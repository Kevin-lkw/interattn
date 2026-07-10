import contextlib

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from .condition_block import (
    _build_prompt_blocks,
    _full_generation_step_metadata,
    _select_prompt_blocks,
    _should_stop,
    _stop_token_ids,
    summarize_condition_block_step_metadata,
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


def _validate_block_sizes(block_sizes):
    sizes = [int(size) for size in block_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("block sizes must be positive")
    if sizes != sorted(sizes, reverse=True):
        raise ValueError("block sizes must be in descending order")
    for parent, child in zip(sizes, sizes[1:]):
        if parent % child != 0:
            raise ValueError("block hierarchy must be nested")
    return sizes


def _condition_block_decode_output_hierarchy(
    *,
    q_grouped,
    pos_tensor,
    prompt_prefixes,
    block_sizes,
    k_suffix,
    v_suffix,
    eps,
    prompt_len,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    k_suffix = k_suffix.float()
    v_suffix = v_suffix.float()

    active = None
    max_parts = []
    cluster_parts = []
    token_parts = []
    stats = {
        "rows": int(n_kv_heads * group_size * n_query),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": 0,
        "total_available": int(((pos_tensor.long() + 1).sum() * n_kv_heads * group_size).item()),
    }

    for level_idx, (block_size, prefix) in enumerate(zip(block_sizes, prompt_prefixes)):
        failed, z_logits, v_bar, size, cluster_exists = _select_prompt_blocks(
            q_grouped,
            prefix,
            eps,
        )
        if active is None:
            active = cluster_exists.view(1, 1, 1, -1).expand_as(failed)
        else:
            active = active[..., : cluster_exists.shape[-1]]
            active = active & cluster_exists.view(1, 1, 1, -1)

        is_leaf = level_idx == len(block_sizes) - 1
        stats["clusters"] += int(active.sum().item())

        accept_cluster = active & ~failed
        if accept_cluster.any():
            cluster_logits = z_logits.masked_fill(~accept_cluster, float("-inf"))
            max_parts.append(cluster_logits.amax(dim=-1))
            cluster_parts.append((cluster_logits, accept_cluster, v_bar))
            stats["hybrid_tokens"] += int(accept_cluster.sum().item())

        failed_active = active & failed
        stats["selected_clusters"] += int(failed_active.sum().item())
        if is_leaf:
            visible = (
                prefix["valid_token"].view(1, 1, 1, -1, block_size)
                & (
                    prefix["token_idx"].view(1, 1, 1, -1, block_size)
                    <= pos_tensor.view(1, 1, -1, 1, 1)
                )
            )
            token_active = failed_active.unsqueeze(-1) & visible
            token_logits = torch.einsum("gsqd,gbtd->gsqbt", q_grouped, prefix["k_block"]) / scale
            token_logits = token_logits.masked_fill(~token_active, float("-inf"))
            max_parts.append(token_logits.flatten(3).amax(dim=-1))
            token_parts.append((token_logits, token_active, prefix["v_block"]))
            selected_tokens = int((failed_active.long() * size.view(1, 1, n_query, -1)).sum().item())
            stats["selected_tokens"] += selected_tokens
            stats["hybrid_tokens"] += selected_tokens
        else:
            ratio = int(block_size // block_sizes[level_idx + 1])
            active = failed_active.repeat_interleave(ratio, dim=-1)

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
        stats["hybrid_tokens"] += int(suffix_active.sum().item())

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    normalizer = torch.zeros_like(max_logit)
    numerator = torch.zeros(
        (*max_logit.shape, head_dim),
        device=q_grouped.device,
        dtype=torch.float32,
    )
    for cluster_logits, cluster_active, v_bar in cluster_parts:
        cluster_exp = torch.exp(cluster_logits - max_logit[:, :, :, None]).masked_fill(
            ~cluster_active,
            0.0,
        )
        normalizer = normalizer + cluster_exp.sum(dim=-1)
        numerator = numerator + (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=3)
    for token_logits, token_active, v_block in token_parts:
        token_exp = torch.exp(token_logits - max_logit[:, :, :, None, None]).masked_fill(
            ~token_active,
            0.0,
        )
        normalizer = normalizer + token_exp.flatten(3).sum(dim=-1)
        numerator = numerator + torch.einsum("gsqbt,gbtd->gsqd", token_exp, v_block)
    if suffix_logits is not None:
        suffix_exp = torch.exp(suffix_logits - max_logit[:, :, :, None]).masked_fill(
            ~suffix_active,
            0.0,
        )
        normalizer = normalizer + suffix_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("grqt,gtd->grqd", suffix_exp, v_suffix)

    output = numerator / normalizer.clamp_min(1e-30).unsqueeze(-1)
    return output, stats


class ConditionBlockHierarchyDecodeRunner:
    def __init__(
        self,
        *,
        model,
        model_config,
        layer_idx_list,
        full_attention_layers,
        block_sizes,
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
        self.block_sizes = _validate_block_sizes(block_sizes)
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
            raise ValueError("condition_block_hierarchy expects batch_size=1 and q_len=1.")
        if self.pos < self.prompt_len:
            raise ValueError("condition_block_hierarchy expects decode positions.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        n_kv_heads = int(key.shape[1])
        if n_heads % n_kv_heads != 0:
            raise ValueError("condition_block_hierarchy expects grouped query attention.")

        q_grouped = query[0].float().reshape(n_kv_heads, n_heads // n_kv_heads, q_len, head_dim)
        k_all = key[0]
        v_all = value[0]
        cache_key = (int(layer_idx), self.prompt_len, tuple(self.block_sizes))
        prompt_prefixes = self.prompt_prefix_cache.get(cache_key)
        if prompt_prefixes is None:
            prompt_prefixes = [
                _build_prompt_blocks(
                    k_all[:, : self.prompt_len],
                    v_all[:, : self.prompt_len],
                    block_size,
                )
                for block_size in self.block_sizes
            ]
            self.prompt_prefix_cache[cache_key] = prompt_prefixes

        output, stats = _condition_block_decode_output_hierarchy(
            q_grouped=q_grouped,
            pos_tensor=torch.tensor([self.pos], device=query.device, dtype=torch.long),
            prompt_prefixes=prompt_prefixes,
            block_sizes=self.block_sizes,
            k_suffix=k_all[:, self.prompt_len :],
            v_suffix=v_all[:, self.prompt_len :],
            eps=self.eps,
            prompt_len=self.prompt_len,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)

        self.stats_by_layer[int(layer_idx)] = stats
        merge_stats(self.aggregate_stats, stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def condition_block_hierarchy_decode_context(runner):
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


def generate_condition_block_hierarchy_cached(
    *,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    block_sizes,
    eps,
    max_new_tokens,
    full_attention_layers=0,
    device=None,
    dataset=None,
):
    del device
    block_sizes = _validate_block_sizes(block_sizes)
    if int(input_ids.shape[0]) != 1:
        raise ValueError("condition_block_hierarchy generate expects batch_size=1.")

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
        runner = ConditionBlockHierarchyDecodeRunner(
            model=model,
            model_config=model.config,
            layer_idx_list=layer_idx_list,
            full_attention_layers=full_attention_layers,
            block_sizes=block_sizes,
            eps=eps,
            prompt_len=prompt_len,
            pos=total_len - 1,
            prompt_prefix_cache=prompt_prefix_cache,
        )
        with condition_block_hierarchy_decode_context(runner):
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
    metadata["condition_block_hierarchy"] = {"block_sizes": block_sizes, "eps": float(eps)}
    return torch.cat(generated, dim=1), metadata
