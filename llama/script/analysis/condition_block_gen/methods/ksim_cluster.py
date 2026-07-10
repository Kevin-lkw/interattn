import contextlib

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from ...condition_block_ppl.condition_ksim_cluster import build_prompt_ksim_clusters
from .condition_block import (
    _full_generation_step_metadata,
    _should_stop,
    _stop_token_ids,
)
from .patching import merge_stats, summarize_stats


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


def _ksim_decode_output(
    *,
    q_grouped,
    pos_tensor,
    prompt_clusters,
    k_suffix,
    v_suffix,
    prompt_len,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    counts = prompt_clusters["counts"].float()
    cluster_exists = counts > 0
    z_logits = torch.einsum("gsqd,gcd->gsqc", q_grouped.float(), prompt_clusters["k_bar"]) / scale
    z_logits = z_logits + torch.log(counts.clamp_min(1.0)).view(n_kv_heads, 1, 1, -1)
    cluster_active = cluster_exists.view(n_kv_heads, 1, 1, -1).expand_as(z_logits)
    z_logits = z_logits.masked_fill(~cluster_active, float("-inf"))

    max_parts = [z_logits.amax(dim=-1)]
    suffix_len = int(k_suffix.shape[1])
    suffix_logits = None
    suffix_active = None
    if suffix_len > 0:
        suffix_pos = torch.arange(
            int(prompt_len),
            int(prompt_len) + suffix_len,
            device=q_grouped.device,
            dtype=torch.long,
        )
        suffix_active = suffix_pos.view(1, 1, 1, -1) <= pos_tensor.view(1, 1, -1, 1)
        suffix_active = suffix_active.expand(n_kv_heads, group_size, -1, -1)
        suffix_logits = torch.einsum("grqd,gtd->grqt", q_grouped.float(), k_suffix.float()) / scale
        suffix_logits = suffix_logits.masked_fill(~suffix_active, float("-inf"))
        max_parts.append(suffix_logits.amax(dim=-1))

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    cluster_exp = torch.exp(z_logits - max_logit[:, :, :, None]).masked_fill(
        ~cluster_active,
        0.0,
    )
    normalizer = cluster_exp.sum(dim=-1)
    v_bar = prompt_clusters["v_bar"][:, None, None].expand(
        n_kv_heads,
        group_size,
        n_query,
        -1,
        head_dim,
    )
    numerator = (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=3)
    if suffix_logits is not None:
        suffix_exp = torch.exp(suffix_logits - max_logit[:, :, :, None]).masked_fill(
            ~suffix_active,
            0.0,
        )
        normalizer = normalizer + suffix_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("grqt,gtd->grqd", suffix_exp, v_suffix.float())

    output = numerator / normalizer.clamp_min(1e-30).unsqueeze(-1)
    visible_clusters = int(cluster_active.sum().item())
    suffix_tokens = int(suffix_active.sum().item()) if suffix_active is not None else 0
    stats = {
        "rows": int(n_kv_heads * group_size * n_query),
        "clusters": visible_clusters,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": int(visible_clusters + suffix_tokens),
        "total_available": int(((pos_tensor.long() + 1).sum() * n_kv_heads * group_size).item()),
    }
    return output, stats


class KSimClusterDecodeRunner:
    def __init__(
        self,
        *,
        model,
        model_config,
        layer_idx_list,
        full_attention_layers,
        cluster_size,
        kmeans_iters,
        prompt_len,
        pos,
        prompt_cluster_cache,
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
        self.prompt_len = int(prompt_len)
        self.pos = int(pos)
        self.prompt_cluster_cache = prompt_cluster_cache
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
            if layer_idx in self.layer_idx_set:
                stats = _full_attention_stats_for_heads(query.shape[1], [self.pos])
                self.stats_by_layer[int(layer_idx)] = stats
                merge_stats(self.aggregate_stats, stats)
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
            raise ValueError("ksim_cluster expects batch_size=1 and q_len=1.")
        if self.pos < self.prompt_len:
            raise ValueError("ksim_cluster expects decode positions.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("ksim_cluster expects grouped query attention.")

        q_grouped = query[0].float().reshape(n_kv_heads, n_heads // n_kv_heads, q_len, head_dim)
        k_all = key[0]
        v_all = value[0]
        cache_key = (
            int(layer_idx),
            self.prompt_len,
            self.cluster_size,
            self.kmeans_iters,
        )
        prompt_clusters = self.prompt_cluster_cache.get(cache_key)
        if prompt_clusters is None:
            prompt_clusters = build_prompt_ksim_clusters(
                k_all[:, : self.prompt_len],
                v_all[:, : self.prompt_len],
                cluster_size=self.cluster_size,
                kmeans_iters=self.kmeans_iters,
            )
            self.prompt_cluster_cache[cache_key] = prompt_clusters

        output, stats = _ksim_decode_output(
            q_grouped=q_grouped,
            pos_tensor=torch.tensor([self.pos], device=query.device, dtype=torch.long),
            prompt_clusters=prompt_clusters,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            prompt_len=self.prompt_len,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)
        self.stats_by_layer[int(layer_idx)] = stats
        merge_stats(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def ksim_cluster_decode_context(runner):
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


def _full_attention_stats_for_heads(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * int(n_heads)
    return {
        "rows": int(n_heads) * len(pos_list),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


def summarize_ksim_cluster_step_metadata(step_metadata):
    aggregate = {}
    by_step = []
    for step_idx, metadata in enumerate(step_metadata):
        if not metadata:
            continue
        step_aggregate = metadata.get("aggregate", {})
        by_step.append(
            {
                "step": step_idx,
                "equiv_budget": step_aggregate.get("mean_budget_causal"),
            }
        )
        for key, value in step_aggregate.items():
            if isinstance(value, int):
                aggregate[key] = int(aggregate.get(key, 0)) + int(value)

    total_available = max(int(aggregate.get("total_available", 0)), 1)
    rows = max(int(aggregate.get("rows", 0)), 1)
    hybrid_tokens = int(aggregate.get("hybrid_tokens", 0))
    equiv_budget = float(hybrid_tokens / total_available)
    return {
        "ksim_cluster_equiv_budget": equiv_budget,
        "ksim_cluster_budget": {
            **aggregate,
            "mean_hybrid_tokens": float(hybrid_tokens / rows),
            "mean_budget_causal": equiv_budget,
            "mean_budget_visible": equiv_budget,
            "by_step": by_step,
        },
    }


def generate_ksim_cluster_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    cluster_size,
    kmeans_iters,
    max_new_tokens,
    full_attention_layers,
    device,
    dataset=None,
):
    if int(input_ids.shape[0]) != 1:
        raise ValueError("ksim_cluster generate currently expects batch_size=1.")

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        output_ids = torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        return output_ids, summarize_ksim_cluster_step_metadata([])

    stop_token_ids = _stop_token_ids(tokenizer, dataset)
    layer_idx_list = list(range(int(model.config.num_hidden_layers)))
    prompt_cluster_cache = {}
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
        return torch.cat(generated, dim=1), summarize_ksim_cluster_step_metadata(step_metadata)

    cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
    step_input_ids = next_id
    total_len += 1

    for _step in range(1, max_new_tokens):
        runner = KSimClusterDecodeRunner(
            model=model,
            model_config=model.config,
            layer_idx_list=layer_idx_list,
            full_attention_layers=full_attention_layers,
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
            prompt_len=prompt_len,
            pos=total_len - 1,
            prompt_cluster_cache=prompt_cluster_cache,
        )
        with ksim_cluster_decode_context(runner):
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

    return torch.cat(generated, dim=1), summarize_ksim_cluster_step_metadata(step_metadata)
