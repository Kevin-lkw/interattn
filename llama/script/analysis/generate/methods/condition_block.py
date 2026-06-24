import contextlib
import math

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from .patching import merge_stats, summarize_stats


def run_prefill_only_condition_block(**_kwargs):
    raise RuntimeError(
        "condition_block now uses the cached generate path only. "
        "Call generate_condition_block_cached through generate_with_method()."
    )


def build_condition_args(method, prompt_len):
    from types import SimpleNamespace

    return SimpleNamespace(
        seq_len=int(prompt_len),
        block_size=method.condition_block_size,
        full_attention_layers=int(method.full_attention_layers),
        delta_mode=method.condition_delta_mode,
    )


def generate_condition_block_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    method,
    device,
    dataset=None,
):
    if method.condition_delta_mode != "range_bound":
        raise ValueError("condition_block only supports delta_mode='range_bound'.")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("condition_block generate currently expects batch_size=1.")

    prompt_len = int(input_ids.shape[1])
    max_new_tokens = int(method.max_new_tokens)
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
        runner = ConditionBlockDecodeRunner(
            model=model,
            model_config=model.config,
            layer_idx_list=layer_idx_list,
            full_attention_layers=method.full_attention_layers,
            block_size=method.condition_block_size,
            eps=method.condition_eps,
            prompt_len=prompt_len,
            pos=total_len - 1,
            prompt_prefix_cache=prompt_prefix_cache,
        )
        with condition_block_decode_context(runner):
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

    return torch.cat(generated, dim=1), summarize_condition_block_step_metadata(step_metadata)


def _stop_token_ids(tokenizer, dataset):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            ids.append(int(newline_ids[-1]))
    return ids


def _should_stop(next_id, stop_token_ids):
    if not stop_token_ids:
        return False
    stop_ids = torch.tensor(stop_token_ids, device=next_id.device)
    return bool(torch.isin(next_id, stop_ids).all().item())


def _pad_blocks(x, block_size):
    n_heads, seq_len = x.shape[:2]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *x.shape[2:]),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *x.shape[2:]), n_blocks


def _build_prompt_blocks(k_all, v_all, block_size):
    k_block, n_blocks = _pad_blocks(k_all.float(), block_size)
    v_block, _ = _pad_blocks(v_all.float(), block_size)
    device = k_all.device
    seq_len = int(k_all.shape[1])

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(n_blocks, block_size)
    valid_token = token_idx < seq_len
    valid = valid_token.view(1, n_blocks, block_size, 1)
    size = valid_token.sum(dim=1).long()
    size_float = size.clamp_min(1).float()

    k_sum = (k_block * valid).sum(dim=2)
    v_sum = (v_block * valid).sum(dim=2)
    k_for_max = k_block.masked_fill(~valid, float("-inf"))
    k_for_min = k_block.masked_fill(~valid, float("inf"))
    v_norm = torch.norm(v_block, p=2, dim=-1)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    return {
        "k_block": k_block,
        "v_block": v_block,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_bar": k_sum / size_float.view(1, n_blocks, 1),
        "v_bar": v_sum / size_float.view(1, n_blocks, 1),
        "k_max": k_for_max.amax(dim=2),
        "k_min": k_for_min.amin(dim=2),
        "v_norm_max": v_norm.amax(dim=2),
        "block_valid_counts": size,
    }


def _select_prompt_blocks(q_grouped, prefix, eps):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    size = prefix["block_valid_counts"].view(1, 1, -1).long()
    cluster_exists = size > 0
    cluster_exists_view = cluster_exists.view(1, 1, 1, -1)
    size_float = size.clamp_min(1).float()

    s_c = torch.einsum("gsqd,gbd->gsqb", q_grouped, prefix["k_bar"]) / scale
    q_bounds = q_grouped[:, :, :, None, :]
    upper = torch.maximum(
        q_bounds * prefix["k_max"][:, None, None],
        q_bounds * prefix["k_min"][:, None, None],
    ).sum(dim=-1) / scale
    lower = torch.minimum(
        q_bounds * prefix["k_max"][:, None, None],
        q_bounds * prefix["k_min"][:, None, None],
    ).sum(dim=-1) / scale
    delta = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
    delta = delta.masked_fill(~cluster_exists_view, 0.0)

    b_c = prefix["v_norm_max"][:, None, None, :].expand(n_kv_heads, group_size, n_query, -1)
    b_c = b_c.masked_fill(~cluster_exists_view, 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).unsqueeze(2) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists_view, float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )

    selected = (condition.mean(dim=1, keepdim=True) > eps) & cluster_exists_view
    selected = selected.expand(n_kv_heads, group_size, n_query, -1)
    v_bar = prefix["v_bar"][:, None, None].expand(
        n_kv_heads,
        group_size,
        n_query,
        -1,
        head_dim,
    )
    return selected, z_logits, v_bar, size.view(1, -1), cluster_exists.view(1, -1)


def _condition_block_decode_output(
    *,
    q_grouped,
    pos_tensor,
    prompt_prefix,
    k_suffix,
    v_suffix,
    block_size,
    eps,
    prompt_len,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    k_suffix = k_suffix.float()
    v_suffix = v_suffix.float()
    selected, z_logits, v_bar, size, cluster_exists = _select_prompt_blocks(
        q_grouped,
        prompt_prefix,
        eps,
    )

    visible = (
        prompt_prefix["valid_token"].view(1, 1, 1, -1, block_size)
        & (
            prompt_prefix["token_idx"].view(1, 1, 1, -1, block_size)
            <= pos_tensor.view(1, 1, -1, 1, 1)
        )
    )
    token_active = selected.unsqueeze(-1) & visible
    token_logits = torch.einsum("gsqd,gbtd->gsqbt", q_grouped, prompt_prefix["k_block"]) / scale
    token_logits = token_logits.masked_fill(~token_active, float("-inf"))

    cluster_active = (~selected) & cluster_exists.view(1, 1, 1, -1)
    cluster_logits = z_logits.masked_fill(~cluster_active, float("-inf"))
    max_parts = [token_logits.flatten(3).amax(dim=-1), cluster_logits.amax(dim=-1)]

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
        suffix_logits = torch.einsum("grqd,gtd->grqt", q_grouped, k_suffix) / scale
        suffix_logits = suffix_logits.masked_fill(~suffix_active, float("-inf"))
        max_parts.append(suffix_logits.amax(dim=-1))

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    token_exp = torch.exp(token_logits - max_logit[:, :, :, None, None]).masked_fill(
        ~token_active,
        0.0,
    )
    cluster_exp = torch.exp(cluster_logits - max_logit[:, :, :, None]).masked_fill(
        ~cluster_active,
        0.0,
    )
    normalizer = token_exp.flatten(3).sum(dim=-1) + cluster_exp.sum(dim=-1)
    numerator = torch.einsum("gsqbt,gbtd->gsqd", token_exp, prompt_prefix["v_block"])
    numerator = numerator + (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=3)

    if suffix_logits is not None:
        suffix_exp = torch.exp(suffix_logits - max_logit[:, :, :, None]).masked_fill(
            ~suffix_active,
            0.0,
        )
        normalizer = normalizer + suffix_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("grqt,gtd->grqd", suffix_exp, v_suffix)

    output = numerator / normalizer.clamp_min(1e-30).unsqueeze(-1)
    stats = _condition_stats(
        selected=selected,
        size=size,
        cluster_exists=cluster_exists,
        pos_tensor=pos_tensor,
        prompt_len=prompt_len,
    )
    return output, stats


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


class ConditionBlockDecodeRunner:
    def __init__(
        self,
        *,
        model,
        model_config,
        layer_idx_list,
        full_attention_layers,
        block_size,
        eps,
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
        self.block_size = int(block_size)
        self.eps = float(eps)
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
            if layer_idx in self.layer_idx_set:
                stats = full_attention_stats_for_heads(query.shape[1], [self.pos])
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
            raise ValueError("condition_block fast path expects batch_size=1 and q_len=1.")
        if self.pos < self.prompt_len:
            raise ValueError("condition_block fast path expects generated-token decode positions.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("condition_block fast path expects grouped query attention.")

        q_grouped = query[0, :, :, :].float().reshape(n_kv_heads, n_heads // n_kv_heads, q_len, head_dim)
        k_all = key[0]
        v_all = value[0]
        cache_key = (int(layer_idx), self.prompt_len, self.block_size)
        prompt_prefix = self.prompt_prefix_cache.get(cache_key)
        if prompt_prefix is None:
            prompt_prefix = _build_prompt_blocks(
                k_all[:, : self.prompt_len],
                v_all[:, : self.prompt_len],
                self.block_size,
            )
            self.prompt_prefix_cache[cache_key] = prompt_prefix

        output, stats = _condition_block_decode_output(
            q_grouped=q_grouped,
            pos_tensor=torch.tensor([self.pos], device=query.device, dtype=torch.long),
            prompt_prefix=prompt_prefix,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            block_size=self.block_size,
            eps=self.eps,
            prompt_len=self.prompt_len,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)

        self.stats_by_layer[int(layer_idx)] = stats
        merge_stats(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def condition_block_decode_context(runner):
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


def _condition_stats(*, selected, size, cluster_exists, pos_tensor, prompt_len):
    n_kv_heads, group_size, n_query = selected.shape[:3]
    selected_tokens = (selected.long() * size.view(1, 1, n_query, -1)).sum()
    cluster_active = (~selected) & cluster_exists.view(1, 1, n_query, -1)
    suffix_tokens = (
        (pos_tensor - int(prompt_len) + 1).clamp_min(0).long().sum()
        * n_kv_heads
        * group_size
    )
    n_rows = int(n_kv_heads * group_size * n_query)
    return {
        "rows": n_rows,
        "clusters": int((cluster_exists.sum() * n_kv_heads * group_size).item()),
        "selected_clusters": int(selected.sum().item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int((selected_tokens + cluster_active.sum() + suffix_tokens).item()),
        "total_available": int(((pos_tensor.long() + 1).sum() * n_kv_heads * group_size).item()),
    }


def full_attention_stats(ctx, pos_list):
    return full_attention_stats_for_heads(int(ctx.model_config.num_attention_heads), pos_list)


def full_attention_stats_for_heads(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * int(n_heads)
    return {
        "rows": int(n_heads) * len(pos_list),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


def _full_generation_step_metadata(model, pos_list):
    aggregate = {}
    by_layer = {}
    n_heads = int(model.config.num_attention_heads)
    for layer_idx in range(int(model.config.num_hidden_layers)):
        stats = full_attention_stats_for_heads(n_heads, pos_list)
        by_layer[layer_idx] = stats
        merge_stats(aggregate, stats)
    return summarize_stats(aggregate, by_layer)


def summarize_condition_block_step_metadata(step_metadata):
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
        "condition_block_equiv_budget": equiv_budget,
        "condition_block_budget": {
            **aggregate,
            "mean_hybrid_tokens": float(hybrid_tokens / rows),
            "mean_budget_causal": equiv_budget,
            "mean_budget_visible": equiv_budget,
            "by_step": by_step,
        },
    }
