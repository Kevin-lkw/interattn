import contextlib

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from ...condition_ksim_cluster import resolve_num_clusters, spherical_kmeans_assign
from .condition_block import (
    _full_generation_step_metadata,
    _should_stop,
    _stop_token_ids,
    summarize_condition_block_step_metadata,
)
from .patching import merge_stats, summarize_stats


def _build_prompt_condition_ksim_prefix(
    k_head,
    v_head,
    *,
    cluster_size,
    kmeans_iters,
):
    if k_head.ndim != 2 or v_head.ndim != 2:
        raise ValueError("k_head and v_head must have shape [prompt_len, head_dim].")
    prompt_len, head_dim = k_head.shape
    n_clusters = resolve_num_clusters(prompt_len, cluster_size)
    assignments = spherical_kmeans_assign(
        k_head,
        cluster_size=cluster_size,
        kmeans_iters=kmeans_iters,
    )
    device = k_head.device
    k_float = k_head.float()
    v_float = v_head.float()
    counts = torch.zeros(n_clusters, device=device, dtype=torch.float32)
    k_sum = torch.zeros(n_clusters, head_dim, device=device, dtype=torch.float32)
    v_sum = torch.zeros(n_clusters, head_dim, device=device, dtype=torch.float32)
    counts.index_add_(
        0,
        assignments,
        torch.ones(prompt_len, device=device, dtype=torch.float32),
    )
    k_sum.index_add_(0, assignments, k_float)
    v_sum.index_add_(0, assignments, v_float)

    k_max = torch.full((n_clusters, head_dim), float("-inf"), device=device, dtype=torch.float32)
    k_min = torch.full((n_clusters, head_dim), float("inf"), device=device, dtype=torch.float32)
    v_norm_max = torch.full((n_clusters,), float("-inf"), device=device, dtype=torch.float32)
    # Prompt generation builds this once per layer/head; a tiny loop over clusters
    # keeps memory bounded for LongBench-length prompts.
    for cluster_idx in range(n_clusters):
        mask = assignments == cluster_idx
        if not mask.any():
            continue
        k_cluster = k_float[mask]
        k_max[cluster_idx] = k_cluster.amax(dim=0)
        k_min[cluster_idx] = k_cluster.amin(dim=0)
        v_norm_max[cluster_idx] = torch.norm(v_float[mask], p=2, dim=-1).amax()

    return {
        "cluster_size": int(cluster_size),
        "n_clusters": n_clusters,
        "assignments": assignments,
        "k_tokens": k_float,
        "v_tokens": v_float,
        "counts": counts,
        "k_sum": k_sum,
        "v_sum": v_sum,
        "k_max": k_max,
        "k_min": k_min,
        "v_norm_max": v_norm_max,
    }


def _prompt_ksim_condition_parts(q_pos, prefix, delta_mode):
    n_heads, n_query, head_dim = q_pos.shape
    scale = head_dim**0.5
    counts = prefix["counts"].float()
    cluster_exists = counts > 0
    counts_safe = counts.clamp_min(1.0)
    k_bar = prefix["k_sum"] / counts_safe[:, None]
    v_bar = prefix["v_sum"] / counts_safe[:, None]
    s_c = torch.einsum("hqd,cd->hqc", q_pos.float(), k_bar) / scale
    token_logits = torch.einsum("hqd,td->hqt", q_pos.float(), prefix["k_tokens"]) / scale
    token_cluster = prefix["assignments"]

    if delta_mode == "exact":
        centered = (token_logits - s_c[:, :, token_cluster]).abs()
        delta_vals = torch.full_like(s_c, float("-inf"))
        for cluster_idx in range(int(prefix["n_clusters"])):
            token_mask = (token_cluster == cluster_idx).view(1, 1, -1)
            values = centered.masked_fill(~token_mask, float("-inf")).amax(dim=-1)
            delta_vals[:, :, cluster_idx] = values
        delta = delta_vals.masked_fill(~cluster_exists.view(1, 1, -1), 0.0)
    elif delta_mode == "range_bound":
        q_for_bounds = q_pos[:, :, None, :]
        upper_score = torch.maximum(
            q_for_bounds * prefix["k_max"],
            q_for_bounds * prefix["k_min"],
        ).sum(dim=-1) / scale
        lower_score = torch.minimum(
            q_for_bounds * prefix["k_max"],
            q_for_bounds * prefix["k_min"],
        ).sum(dim=-1) / scale
        delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.view(1, 1, -1), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = prefix["v_norm_max"].masked_fill(~cluster_exists, 0.0).view(1, 1, -1)
    b_all = b_c.amax(dim=-1)
    z_logits = torch.log(counts_safe).view(1, 1, -1) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.view(1, 1, -1), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    return {
        "condition": condition,
        "z_logits": z_logits,
        "v_bar": v_bar.view(1, 1, prefix["n_clusters"], -1).expand(
            n_heads,
            n_query,
            -1,
            -1,
        ),
        "counts": counts.long().view(1, -1).expand(n_query, -1),
        "cluster_exists": cluster_exists.view(1, -1).expand(n_query, -1),
        "token_logits": token_logits,
        "token_cluster": token_cluster,
    }


def _sdpa_full_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **_kwargs,
):
    dropout_p = float(dropout) if module.training else 0.0
    is_full_sequence_causal = query.shape[2] == key.shape[2]
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None if is_full_sequence_causal else attention_mask,
        dropout_p=dropout_p,
        is_causal=is_full_sequence_causal,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _condition_ksim_decode_one_kv_head(
    *,
    q_pos,
    prompt_prefix,
    k_suffix,
    v_suffix,
    prompt_len,
    pos_tensor,
    eps,
    delta_mode,
):
    n_heads, n_query, head_dim = q_pos.shape
    scale = head_dim**0.5
    parts = _prompt_ksim_condition_parts(q_pos, prompt_prefix, delta_mode)
    cluster_exists = parts["cluster_exists"]
    failed = (
        (parts["condition"].mean(dim=0, keepdim=True) > eps)
        & cluster_exists.unsqueeze(0)
    ).expand(n_heads, -1, -1)
    active = cluster_exists.unsqueeze(0).expand_as(failed)
    accept_cluster = active & ~failed
    failed_cluster = active & failed

    max_parts = []
    cluster_logits = None
    if accept_cluster.any():
        cluster_logits = parts["z_logits"].masked_fill(~accept_cluster, float("-inf"))
        max_parts.append(cluster_logits.amax(dim=-1))

    token_active = failed_cluster[:, :, parts["token_cluster"]]
    token_logits = None
    if token_active.any():
        token_logits = parts["token_logits"].masked_fill(~token_active, float("-inf"))
        max_parts.append(token_logits.amax(dim=-1))

    suffix_len = int(k_suffix.shape[0])
    suffix_logits = None
    suffix_active = None
    if suffix_len > 0:
        suffix_pos = torch.arange(
            int(prompt_len),
            int(prompt_len) + suffix_len,
            device=q_pos.device,
            dtype=torch.long,
        )
        suffix_active = suffix_pos.view(1, 1, -1) <= pos_tensor.view(1, n_query, 1)
        suffix_active = suffix_active.expand(n_heads, -1, -1)
        suffix_logits = torch.einsum("hqd,td->hqt", q_pos.float(), k_suffix.float()) / scale
        suffix_logits = suffix_logits.masked_fill(~suffix_active, float("-inf"))
        max_parts.append(suffix_logits.amax(dim=-1))

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    normalizer = torch.zeros_like(max_logit)
    numerator = torch.zeros(
        (*max_logit.shape, head_dim),
        device=q_pos.device,
        dtype=torch.float32,
    )
    if cluster_logits is not None:
        cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
            ~accept_cluster,
            0.0,
        )
        normalizer = normalizer + cluster_exp.sum(dim=-1)
        numerator = numerator + (cluster_exp.unsqueeze(-1) * parts["v_bar"]).sum(dim=2)
    if token_logits is not None:
        token_exp = torch.exp(token_logits - max_logit[:, :, None]).masked_fill(
            ~token_active,
            0.0,
        )
        normalizer = normalizer + token_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("hqt,td->hqd", token_exp, prompt_prefix["v_tokens"])
    suffix_tokens = 0
    if suffix_logits is not None:
        suffix_exp = torch.exp(suffix_logits - max_logit[:, :, None]).masked_fill(
            ~suffix_active,
            0.0,
        )
        normalizer = normalizer + suffix_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("hqt,td->hqd", suffix_exp, v_suffix.float())
        suffix_tokens = int(suffix_active.sum().item())

    selected_tokens = int(
        (failed_cluster.long() * parts["counts"].view(1, n_query, -1)).sum().item()
    )
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int(active.sum().item()),
        "selected_clusters": int(failed_cluster.sum().item()),
        "selected_tokens": selected_tokens + suffix_tokens,
        "hybrid_tokens": int(accept_cluster.sum().item()) + selected_tokens + suffix_tokens,
        "total_available": int(((pos_tensor.long() + 1).sum() * n_heads).item()),
    }
    return numerator / normalizer.clamp_min(1e-30).unsqueeze(-1), stats


def _condition_ksim_decode_output(
    *,
    q_grouped,
    pos_tensor,
    prompt_prefixes,
    k_suffix,
    v_suffix,
    prompt_len,
    eps,
    delta_mode,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    output = torch.empty(
        n_kv_heads,
        group_size,
        n_query,
        head_dim,
        device=q_grouped.device,
        dtype=torch.float32,
    )
    stats = {}
    for kv_head in range(n_kv_heads):
        head_output, head_stats = _condition_ksim_decode_one_kv_head(
            q_pos=q_grouped[kv_head],
            prompt_prefix=prompt_prefixes[kv_head],
            k_suffix=k_suffix[kv_head],
            v_suffix=v_suffix[kv_head],
            prompt_len=prompt_len,
            pos_tensor=pos_tensor,
            eps=eps,
            delta_mode=delta_mode,
        )
        output[kv_head] = head_output
        merge_stats(stats, head_stats)
    return output, stats


class ConditionKSimClusterDecodeRunner:
    def __init__(
        self,
        *,
        model,
        model_config,
        layer_idx_list,
        full_attention_layers,
        cluster_size,
        kmeans_iters,
        eps,
        delta_mode,
        prompt_len,
        pos,
        prompt_prefix_cache,
    ):
        self.model_config = model_config
        self.module_to_layer_idx = {
            id(layer.self_attn): int(layer_idx)
            for layer_idx, layer in enumerate(model.model.layers)
        }
        self.layer_idx_set = {int(layer_idx) for layer_idx in layer_idx_list}
        self.full_attention_layers = int(full_attention_layers)
        self.cluster_size = int(cluster_size)
        self.kmeans_iters = int(kmeans_iters)
        self.eps = float(eps)
        self.delta_mode = str(delta_mode)
        self.prompt_len = int(prompt_len)
        self.pos = int(pos)
        self.prompt_prefix_cache = prompt_prefix_cache
        self.stats_by_layer = {}
        self.aggregate_stats = {}

    def should_compress(self, layer_idx):
        return int(layer_idx) in self.layer_idx_set and int(layer_idx) >= self.full_attention_layers

    def summarize(self):
        return summarize_stats(self.aggregate_stats, self.stats_by_layer)

    def hybrid_attention_forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            layer_idx = self.module_to_layer_idx.get(id(module))

        if layer_idx is None or not self.should_compress(layer_idx):
            return _sdpa_full_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )

        if query.shape[0] != 1 or query.shape[2] != 1:
            raise ValueError("condition_ksim_cluster expects batch_size=1 and q_len=1.")
        if self.pos < self.prompt_len:
            raise ValueError("condition_ksim_cluster expects decode positions.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("condition_ksim_cluster expects grouped query attention.")

        q_grouped = query[0].float().reshape(n_kv_heads, n_heads // n_kv_heads, q_len, head_dim)
        k_all = key[0]
        v_all = value[0]
        cache_key = (
            int(layer_idx),
            self.prompt_len,
            self.cluster_size,
            self.kmeans_iters,
        )
        prompt_prefixes = self.prompt_prefix_cache.get(cache_key)
        if prompt_prefixes is None:
            prompt_prefixes = [
                _build_prompt_condition_ksim_prefix(
                    k_all[kv_head, : self.prompt_len],
                    v_all[kv_head, : self.prompt_len],
                    cluster_size=self.cluster_size,
                    kmeans_iters=self.kmeans_iters,
                )
                for kv_head in range(n_kv_heads)
            ]
            self.prompt_prefix_cache[cache_key] = prompt_prefixes

        output, stats = _condition_ksim_decode_output(
            q_grouped=q_grouped,
            pos_tensor=torch.tensor([self.pos], device=query.device, dtype=torch.long),
            prompt_prefixes=prompt_prefixes,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            prompt_len=self.prompt_len,
            eps=self.eps,
            delta_mode=self.delta_mode,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)
        self.stats_by_layer[int(layer_idx)] = stats
        merge_stats(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def condition_ksim_cluster_decode_context(runner):
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = runner.hybrid_attention_forward
    try:
        yield runner
    finally:
        modeling_llama.eager_attention_forward = original_eager


@contextlib.contextmanager
def full_attention_sdpa_context():
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = _sdpa_full_attention_forward
    try:
        yield
    finally:
        modeling_llama.eager_attention_forward = original_eager


def generate_condition_ksim_cluster_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    cluster_size,
    kmeans_iters,
    eps,
    delta_mode,
    max_new_tokens,
    full_attention_layers=0,
    device=None,
    dataset=None,
):
    del device
    if int(input_ids.shape[0]) != 1:
        raise ValueError("condition_ksim_cluster generate expects batch_size=1.")

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        output_ids = torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        return output_ids, summarize_condition_block_step_metadata([])

    stop_token_ids = _stop_token_ids(tokenizer, dataset)
    layer_idx_list = list(range(int(model.config.num_hidden_layers)))
    prompt_prefix_cache = {}
    generated = []
    step_metadata = []
    cur_mask = attention_mask
    total_len = prompt_len

    with full_attention_sdpa_context():
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=cur_mask,
                use_cache=True,
                logits_to_keep=1,
            )

    logits = outputs.logits.float()
    past_key_values = outputs.past_key_values
    step_metadata.append(_full_generation_step_metadata(model, [total_len - 1]))

    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_id)
    if _should_stop(next_id, stop_token_ids):
        return torch.cat(generated, dim=1), summarize_condition_block_step_metadata(step_metadata)

    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
    step_input_ids = next_id
    total_len += 1

    for _step in range(1, max_new_tokens):
        runner = ConditionKSimClusterDecodeRunner(
            model=model,
            model_config=model.config,
            layer_idx_list=layer_idx_list,
            full_attention_layers=full_attention_layers,
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
            eps=eps,
            delta_mode=delta_mode,
            prompt_len=prompt_len,
            pos=total_len - 1,
            prompt_prefix_cache=prompt_prefix_cache,
        )
        with condition_ksim_cluster_decode_context(runner):
            with torch.no_grad():
                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )

        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values
        step_metadata.append(runner.summarize())

        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        if _should_stop(next_id, stop_token_ids):
            break

        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        step_input_ids = next_id
        total_len += 1

    metadata = summarize_condition_block_step_metadata(step_metadata)
    metadata["condition_ksim_cluster"] = {
        "cluster_size": int(cluster_size),
        "kmeans_iters": int(kmeans_iters),
        "eps": float(eps),
        "delta_mode": str(delta_mode),
    }
    return torch.cat(generated, dim=1), metadata
