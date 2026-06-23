import contextlib
import math

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from ...sanity import grouped_query_heads
from .patching import merge_stats, run_with_prefill_only_patches, summarize_stats


def run_prefill_only_condition_block(
    *,
    ctx,
    args,
    eps,
    layer_idx_list,
    prompt_len,
    pos_list,
    model_inputs,
):
    if args.delta_mode == "range_bound":
        return run_prefill_only_condition_block_optim(
            ctx=ctx,
            args=args,
            eps=eps,
            layer_idx_list=layer_idx_list,
            prompt_len=prompt_len,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
    return run_prefill_only_condition_block_patch(
        ctx=ctx,
        args=args,
        eps=eps,
        layer_idx_list=layer_idx_list,
        prompt_len=prompt_len,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )


def run_prefill_only_condition_block_patch(
    *,
    ctx,
    args,
    eps,
    layer_idx_list,
    prompt_len,
    pos_list,
    model_inputs,
):
    def _build_patch(layer_ctx, layer_idx, artifacts):
        return build_prefill_only_condition_block_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            prompt_len=prompt_len,
            block_size=args.block_size,
            eps=eps,
            delta_mode=args.delta_mode,
        )

    return run_with_prefill_only_patches(
        ctx=ctx,
        layer_idx_list=layer_idx_list,
        pos_list=pos_list,
        model_inputs=model_inputs,
        full_attention_layers=args.full_attention_layers,
        build_patch=_build_patch,
        collect_stats=True,
        full_attention_stats=full_attention_stats(ctx, pos_list),
    )


def run_prefill_only_condition_block_optim(
    *,
    ctx,
    args,
    eps,
    layer_idx_list,
    prompt_len,
    pos_list,
    model_inputs,
):
    seq_len = int(model_inputs["input_ids"].shape[1])
    tail_len = len(pos_list)
    use_tail_logits = pos_list == list(range(seq_len - tail_len, seq_len))
    runner = ConditionBlockGenerateOptimForward(
        model=ctx.model,
        model_config=ctx.model_config,
        layer_idx_list=layer_idx_list,
        full_attention_layers=args.full_attention_layers,
        block_size=args.block_size,
        eps=eps,
        prompt_len=prompt_len,
        pos_list=pos_list,
    )
    with condition_block_generate_optim_context(runner):
        with torch.no_grad():
            outputs = ctx.model(
                **model_inputs,
                use_cache=False,
                logits_to_keep=tail_len if use_tail_logits else 0,
            )
            logits = outputs.logits if use_tail_logits else outputs.logits[:, pos_list, :]
            logits = logits.float()
    return logits, runner.summarize()


def build_condition_args(method, prompt_len):
    from types import SimpleNamespace

    args = SimpleNamespace(
        seq_len=prompt_len,
        block_size=method.condition_block_size,
        full_attention_layers=method.full_attention_layers,
        delta_mode=method.condition_delta_mode,
    )
    return args


def _pad_blocks(x, block_size):
    n_heads, seq_len = x.shape[:2]
    tail_shape = x.shape[2:]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *tail_shape),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *tail_shape), n_blocks


def _gather_prefix(prefix_tensor, prefix_idx):
    n_heads, n_blocks, block_size = prefix_tensor.shape[:3]
    tail_shape = prefix_tensor.shape[3:]
    n_query = prefix_idx.shape[0]
    expanded = prefix_tensor.unsqueeze(1).expand(
        n_heads, n_query, n_blocks, block_size, *tail_shape
    )
    gather_idx = prefix_idx.view(1, n_query, n_blocks, 1, *([1] * len(tail_shape))).expand(
        n_heads, n_query, n_blocks, 1, *tail_shape
    )
    return torch.gather(expanded, dim=3, index=gather_idx).squeeze(3)


def _build_block_prefix_tensors(k_all, v_all, block_size):
    k_block, n_blocks = _pad_blocks(k_all.float(), block_size)
    v_block, _ = _pad_blocks(v_all.float(), block_size)
    device = k_all.device
    seq_len = k_all.shape[1]

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(
        n_blocks, block_size
    )
    valid_token = token_idx < seq_len

    valid_k = valid_token.view(1, n_blocks, block_size, 1)
    k_for_max = k_block.masked_fill(~valid_k, float("-inf"))
    k_for_min = k_block.masked_fill(~valid_k, float("inf"))
    v_norm = torch.norm(v_block, p=2, dim=-1)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    return {
        "k_block": k_block,
        "v_block": v_block,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_cumsum": k_block.cumsum(dim=2),
        "v_cumsum": v_block.cumsum(dim=2),
        "k_prefix_max": k_for_max.cummax(dim=2).values,
        "k_prefix_min": k_for_min.cummin(dim=2).values,
        "v_norm_prefix_max": v_norm.cummax(dim=2).values,
        "block_starts": torch.arange(n_blocks, device=device) * block_size,
        "block_valid_counts": valid_token.sum(dim=1),
    }


def _range_bound_selection_and_summaries(
    *,
    q_pos,
    pos_tensor,
    prefix,
    block_size,
    eps,
    prompt_len,
):
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = head_dim**0.5

    prompt_visible_len = torch.minimum(
        pos_tensor + 1,
        torch.full_like(pos_tensor, int(prompt_len)),
    )
    raw_prefix_len = prompt_visible_len[:, None] - prefix["block_starts"][None, :]
    prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
    size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
    cluster_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)
    size_float = size.clamp_min(1).float()

    k_sum = _gather_prefix(prefix["k_cumsum"], prefix_idx)
    v_sum = _gather_prefix(prefix["v_cumsum"], prefix_idx)
    k_bar = k_sum / size_float.view(1, n_query, n_blocks, 1)
    v_bar = v_sum / size_float.view(1, n_query, n_blocks, 1)

    s_c = (q_pos[:, :, None, :] * k_bar).sum(dim=-1) / scale
    k_max = _gather_prefix(prefix["k_prefix_max"], prefix_idx)
    k_min = _gather_prefix(prefix["k_prefix_min"], prefix_idx)
    q_for_bounds = q_pos[:, :, None, :]
    upper_score = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
        dim=-1
    ) / scale
    lower_score = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
        dim=-1
    ) / scale
    delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
    delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)

    b_c = _gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
    selected = selected.expand(n_heads, -1, -1)
    return selected, z_logits, v_bar, size, cluster_exists


def _online_update(current_m, current_l, current_o, logits, values, active):
    logits = logits.masked_fill(~active, float("-inf"))
    block_m = logits.amax(dim=-1)
    has_any = active.any(dim=-1)
    new_m = torch.maximum(current_m, block_m)
    new_m = torch.where(has_any, new_m, current_m)

    old_scale = torch.exp(current_m - new_m)
    old_scale = torch.where(torch.isfinite(current_m), old_scale, torch.zeros_like(old_scale))
    exp_logits = torch.exp(logits - new_m.unsqueeze(-1)).masked_fill(~active, 0.0)
    block_l = exp_logits.sum(dim=-1)
    new_l = current_l * old_scale + block_l
    block_o = (exp_logits.unsqueeze(-1) * values).sum(dim=-2)
    new_o = current_o * old_scale.unsqueeze(-1) + block_o
    return new_m, new_l, new_o


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
    sdpa_mask = None if is_full_sequence_causal else attention_mask
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=sdpa_mask,
        dropout_p=dropout_p,
        is_causal=is_full_sequence_causal,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _stats_from_selection(*, selected, size, cluster_exists, pos_tensor, prompt_len, stats_mask):
    selected_stats = selected[:, stats_mask]
    size_stats = size[stats_mask]
    cluster_exists_stats = cluster_exists[stats_mask]
    pos_stats = pos_tensor[stats_mask]
    n_heads = selected.shape[0]
    n_query = int(pos_stats.numel())
    selected_tokens = (selected_stats.long() * size_stats.view(1, n_query, -1)).sum()
    cluster_active = (~selected_stats) & cluster_exists_stats.unsqueeze(0)
    suffix_tokens = (pos_stats - int(prompt_len) + 1).clamp_min(0).long().sum() * n_heads
    return {
        "rows": int(n_heads * n_query),
        "clusters": int((cluster_exists_stats.sum() * n_heads).item()),
        "selected_clusters": int(selected_stats.sum().item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int((selected_tokens + cluster_active.sum() + suffix_tokens).item()),
        "total_available": int(((pos_stats.long() + 1).sum() * n_heads).item()),
    }


def _streaming_generate_hybrid_outputs(
    *,
    q_pos,
    pos_tensor,
    prompt_prefix,
    k_suffix,
    v_suffix,
    block_size,
    eps,
    prompt_len,
    stats_mask,
):
    n_heads, n_query, head_dim = q_pos.shape
    n_prompt_blocks = prompt_prefix["block_starts"].numel()
    scale = head_dim**0.5
    selected, z_logits, v_bar, size, cluster_exists = _range_bound_selection_and_summaries(
        q_pos=q_pos,
        pos_tensor=pos_tensor,
        prefix=prompt_prefix,
        block_size=block_size,
        eps=eps,
        prompt_len=prompt_len,
    )

    running_m = torch.full(
        (n_heads, n_query), float("-inf"), device=q_pos.device, dtype=torch.float32
    )
    running_l = torch.zeros((n_heads, n_query), device=q_pos.device, dtype=torch.float32)
    running_o = torch.zeros(
        (n_heads, n_query, head_dim), device=q_pos.device, dtype=torch.float32
    )

    for block_idx in range(n_prompt_blocks):
        visible = (
            prompt_prefix["valid_token"][block_idx].view(1, -1)
            & (prompt_prefix["token_idx"][block_idx].view(1, -1) <= pos_tensor.view(-1, 1))
        )
        token_active = selected[:, :, block_idx].unsqueeze(-1) & visible.unsqueeze(0)
        if bool(token_active.any().item()):
            k_block = prompt_prefix["k_block"][:, block_idx]
            v_block = prompt_prefix["v_block"][:, block_idx]
            token_logits = torch.einsum("hqd,htd->hqt", q_pos, k_block) / scale
            token_values = v_block.unsqueeze(1).expand(n_heads, n_query, -1, -1)
            running_m, running_l, running_o = _online_update(
                running_m,
                running_l,
                running_o,
                token_logits,
                token_values,
                token_active,
            )

        cluster_active = (~selected[:, :, block_idx]) & cluster_exists[:, block_idx].unsqueeze(0)
        if bool(cluster_active.any().item()):
            cluster_logits = z_logits[:, :, block_idx].unsqueeze(-1)
            cluster_values = v_bar[:, :, block_idx].unsqueeze(-2)
            running_m, running_l, running_o = _online_update(
                running_m,
                running_l,
                running_o,
                cluster_logits,
                cluster_values,
                cluster_active.unsqueeze(-1),
            )

    suffix_len = int(k_suffix.shape[1])
    if suffix_len > 0:
        suffix_pos = torch.arange(
            int(prompt_len),
            int(prompt_len) + suffix_len,
            device=q_pos.device,
            dtype=torch.long,
        )
        for start in range(0, suffix_len, block_size):
            end = min(start + block_size, suffix_len)
            token_idx = suffix_pos[start:end]
            active = token_idx.view(1, -1) <= pos_tensor.view(-1, 1)
            token_active = active.unsqueeze(0).expand(n_heads, -1, -1)
            if bool(token_active.any().item()):
                k_block = k_suffix[:, start:end]
                v_block = v_suffix[:, start:end]
                token_logits = torch.einsum("hqd,htd->hqt", q_pos, k_block) / scale
                token_values = v_block.unsqueeze(1).expand(n_heads, n_query, -1, -1)
                running_m, running_l, running_o = _online_update(
                    running_m,
                    running_l,
                    running_o,
                    token_logits,
                    token_values,
                    token_active,
                )

    output = running_o / running_l.clamp_min(1e-30).unsqueeze(-1)
    stats = _stats_from_selection(
        selected=selected,
        size=size,
        cluster_exists=cluster_exists,
        pos_tensor=pos_tensor,
        prompt_len=prompt_len,
        stats_mask=stats_mask,
    )
    return output, stats


class ConditionBlockGenerateOptimForward:
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
        pos_list,
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
        self.pos_list = [int(pos) for pos in pos_list]
        self.stats_by_layer = {}
        self.aggregate_stats = {}

    def should_compress(self, layer_idx):
        return int(layer_idx) in self.layer_idx_set and int(layer_idx) >= self.full_attention_layers

    def record_full_layer(self, layer_idx, n_heads):
        if layer_idx is None or int(layer_idx) not in self.layer_idx_set:
            return
        layer_idx = int(layer_idx)
        if layer_idx in self.stats_by_layer:
            return
        stats = full_attention_stats_for_heads(n_heads, self.pos_list)
        self.stats_by_layer[layer_idx] = stats
        merge_stats(self.aggregate_stats, stats)

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
            self.record_full_layer(layer_idx, query.shape[1])
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

        if query.shape[0] != 1:
            raise ValueError("condition_block generate optim currently expects batch_size=1.")
        if query.shape[2] != key.shape[2]:
            raise ValueError("condition_block generate optim expects use_cache=False.")
        if attention_mask is not None and attention_mask.shape[-1] != key.shape[2]:
            raise ValueError("Unsupported attention_mask shape for condition_block generate optim.")

        _batch_size, n_heads, q_len, head_dim = query.shape
        if self.prompt_len > q_len:
            raise ValueError(
                f"prompt_len={self.prompt_len} cannot exceed current sequence length {q_len}."
            )
        output, _ = _sdpa_full_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        output_dtype = output.dtype
        pos_tensor = torch.tensor(self.pos_list, device=query.device, dtype=torch.long)
        if bool((pos_tensor < 0).any().item()) or bool((pos_tensor >= q_len).any().item()):
            raise ValueError(f"pos_list={self.pos_list} is out of range for q_len={q_len}.")
        stats_mask = torch.ones(pos_tensor.numel(), device=query.device, dtype=torch.bool)
        layer_stats = {}
        for kv_head, _out_indices, query_heads in grouped_query_heads(
            list(range(n_heads)),
            self.model_config,
            num_kv_heads=key.shape[1],
        ):
            q_pos = query[0, query_heads][:, pos_tensor, :].float()
            k_group = key[0, kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            v_group = value[0, kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            prompt_prefix = _build_block_prefix_tensors(
                k_group[:, : self.prompt_len],
                v_group[:, : self.prompt_len],
                self.block_size,
            )
            group_output, group_stats = _streaming_generate_hybrid_outputs(
                q_pos=q_pos,
                pos_tensor=pos_tensor,
                prompt_prefix=prompt_prefix,
                k_suffix=k_group[:, self.prompt_len :],
                v_suffix=v_group[:, self.prompt_len :],
                block_size=self.block_size,
                eps=self.eps,
                prompt_len=self.prompt_len,
                stats_mask=stats_mask,
            )
            for head_offset, query_head in enumerate(query_heads):
                output[0, pos_tensor, query_head, :] = group_output[head_offset].to(output_dtype)
            merge_stats(layer_stats, group_stats)

        self.stats_by_layer[int(layer_idx)] = layer_stats
        merge_stats(self.aggregate_stats, layer_stats)
        return output, None


@contextlib.contextmanager
def condition_block_generate_optim_context(runner):
    original_eager = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = runner.hybrid_attention_forward
    try:
        yield runner
    finally:
        modeling_llama.eager_attention_forward = original_eager


def build_prefill_only_condition_block_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    prompt_len,
    block_size,
    eps,
    delta_mode,
):
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    output_dtype = artifacts["attn_output"].dtype

    n_heads = q_all.shape[0]
    n_pos = len(pos_list)
    pos_tensor = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    output = torch.empty(
        n_heads,
        n_pos,
        q_all.shape[-1],
        device=ctx.device,
        dtype=torch.float32,
    )
    stats = {
        "rows": 0,
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": 0,
        "total_available": 0,
    }

    for kv_head, _out_indices, query_heads in grouped_query_heads(
        list(range(n_heads)),
        ctx.model_config,
        num_kv_heads=k_all.shape[0],
    ):
        k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1).float()
        v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1).float()
        for local_pos, pos in enumerate(pos_tensor.tolist()):
            prefix_len = min(prompt_len, int(pos) + 1)
            suffix_start = prompt_len
            suffix_end = int(pos) + 1
            head_conditions = []
            for head_offset, query_head in enumerate(query_heads):
                head_conditions.append(
                    condition_tensor_for_prompt_blocks(
                        q_all[query_head, pos].float(),
                        k_group[head_offset, :prefix_len],
                        v_group[head_offset, :prefix_len],
                        block_size,
                        delta_mode,
                    )
                )
            selected_blocks = torch.stack(head_conditions, dim=0).mean(dim=0) > eps

            for head_offset, query_head in enumerate(query_heads):
                q = q_all[query_head, pos].float()
                logits_parts = []
                value_parts = []
                block_idx = 0
                head_hybrid_tokens = 0
                head_selected_clusters = 0
                head_selected_tokens = 0
                for start in range(0, prefix_len, block_size):
                    end = min(start + block_size, prefix_len)
                    k_block = k_group[head_offset, start:end]
                    v_block = v_group[head_offset, start:end]
                    if bool(selected_blocks[block_idx].item()):
                        logits_parts.append(torch.mv(k_block, q) / math.sqrt(q.numel()))
                        value_parts.append(v_block)
                        head_hybrid_tokens += end - start
                        head_selected_clusters += 1
                        head_selected_tokens += end - start
                    else:
                        k_bar = k_block.mean(dim=0)
                        v_bar = v_block.mean(dim=0, keepdim=True)
                        score = torch.dot(q, k_bar) / math.sqrt(q.numel())
                        logits_parts.append((math.log(end - start) + score).reshape(1))
                        value_parts.append(v_bar)
                        head_hybrid_tokens += 1
                    block_idx += 1

                if suffix_end > suffix_start:
                    k_suffix = k_group[head_offset, suffix_start:suffix_end]
                    v_suffix = v_group[head_offset, suffix_start:suffix_end]
                    logits_parts.append(torch.mv(k_suffix, q) / math.sqrt(q.numel()))
                    value_parts.append(v_suffix)
                    head_hybrid_tokens += suffix_end - suffix_start

                weights = torch.softmax(torch.cat(logits_parts, dim=0), dim=0)
                values = torch.cat(value_parts, dim=0)
                output[query_head, local_pos] = (weights.unsqueeze(-1) * values).sum(dim=0)
                stats["rows"] += 1
                stats["clusters"] += int(selected_blocks.numel())
                stats["selected_clusters"] += int(head_selected_clusters)
                stats["selected_tokens"] += int(head_selected_tokens)
                stats["hybrid_tokens"] += int(head_hybrid_tokens)
                stats["total_available"] += int(pos) + 1

    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.to(output_dtype)
        .permute(1, 0, 2)
        .reshape(n_pos, -1)
        .to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach(), stats


def full_attention_stats(ctx, pos_list):
    n_heads = int(ctx.model_config.num_attention_heads)
    return full_attention_stats_for_heads(n_heads, pos_list)


def full_attention_stats_for_heads(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * n_heads
    return {
        "rows": n_heads * len(pos_list),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


def condition_tensor_for_prompt_blocks(q, k_head, v_head, block_size, delta_mode):
    scale = math.sqrt(q.numel())
    seq_len = k_head.shape[0]
    s_vals = []
    deltas = []
    b_cs = []
    size_vals = []
    b_all = torch.norm(v_head.float(), p=2, dim=-1).max()

    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        k_cluster = k_head[start:end].float()
        v_cluster = v_head[start:end].float()
        k_bar = k_cluster.mean(dim=0)
        s_c = torch.dot(q.float(), k_bar.float()) / scale
        if delta_mode == "exact":
            qk_cluster = torch.mv(k_cluster, q.float()) / scale
            delta = (qk_cluster - s_c).abs().max()
        elif delta_mode == "range_bound":
            k_max = k_cluster.max(dim=0).values
            k_min = k_cluster.min(dim=0).values
            upper_score = torch.maximum(q.float() * k_max, q.float() * k_min).sum() / scale
            lower_score = torch.minimum(q.float() * k_max, q.float() * k_min).sum() / scale
            delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        else:
            raise ValueError(f"Unknown delta mode: {delta_mode}")
        s_vals.append(s_c)
        deltas.append(delta)
        b_cs.append(torch.norm(v_cluster, p=2, dim=-1).max())
        size_vals.append(float(end - start))

    s_tensor = torch.stack(s_vals).float()
    size_tensor = torch.tensor(size_vals, device=s_tensor.device, dtype=torch.float32)
    delta_tensor = torch.stack(deltas).float()
    b_c_tensor = torch.stack(b_cs).float()
    p_tensor = torch.softmax(torch.log(size_tensor) + s_tensor, dim=0)
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    return p_tensor * (
        2.0 * b_all * (torch.cosh(delta_tensor) - 1.0) / denom
        + 2.0 * b_c_tensor * torch.tanh(delta_tensor / 2.0)
    )
