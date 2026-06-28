import contextlib
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.llama import modeling_llama
from transformers.integrations.sdpa_attention import sdpa_attention_forward

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

    if os.environ.get("CONDITION_BLOCK_HF_GENERATE_LOOP") == "1":
        return _generate_condition_block_with_hf_loop(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            method=method,
            dataset=dataset,
        )

    stop_token_ids = _stop_token_ids(tokenizer, dataset)
    stop_ids_tensor = (
        torch.tensor(stop_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        if stop_token_ids
        else None
    )
    layer_idx_list = list(range(int(model.config.num_hidden_layers)))
    prompt_prefix_cache = {}
    generated = []
    step_metadata = []
    collect_stats = os.environ.get("CONDITION_BLOCK_SKIP_STATS") != "1"
    cur_mask = attention_mask
    total_len = prompt_len

    # Use Transformers' native SDPA interface for the long prompt. This keeps
    # prefill identical to the optimized full-attention baseline; the custom
    # eager hook is installed only for single-token decode below.
    with model_attention_implementation(model, "sdpa"):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=cur_mask,
                use_cache=True,
                logits_to_keep=1,
            )

    logits = outputs.logits.float()
    past_key_values = outputs.past_key_values
    if collect_stats:
        step_metadata.append(_full_generation_step_metadata(model, [total_len - 1]))

    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_id)
    if _should_stop(next_id, stop_ids_tensor):
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
        if collect_stats:
            step_metadata.append(runner.summarize())

        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_id)
        if _should_stop(next_id, stop_ids_tensor):
            break

        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)
        step_input_ids = next_id
        total_len += 1

    return torch.cat(generated, dim=1), summarize_condition_block_step_metadata(step_metadata)


def _generate_condition_block_with_hf_loop(
    *, model, tokenizer, input_ids, attention_mask, method, dataset
):
    prompt_len = int(input_ids.shape[1])
    runner = ConditionBlockDecodeRunner(
        model=model,
        model_config=model.config,
        layer_idx_list=list(range(int(model.config.num_hidden_layers))),
        full_attention_layers=method.full_attention_layers,
        block_size=method.condition_block_size,
        eps=method.condition_eps,
        prompt_len=prompt_len,
        pos=prompt_len - 1,
        prompt_prefix_cache={},
    )
    generate_kwargs = {}
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_ids[-1]]
    with condition_block_decode_context(runner), torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(method.max_new_tokens),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    return output_ids[:, prompt_len:], {"condition_block_stats_disabled": True}


def _stop_token_ids(tokenizer, dataset):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if dataset == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            ids.append(int(newline_ids[-1]))
    return ids


def _should_stop(next_id, stop_ids_tensor):
    if stop_ids_tensor is None:
        return False
    return bool(torch.isin(next_id, stop_ids_tensor).all().item())


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
    k_block_attn, n_blocks = _pad_blocks(k_all, block_size)
    v_block_attn, _ = _pad_blocks(v_all, block_size)
    device = k_all.device
    seq_len = int(k_all.shape[1])

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(n_blocks, block_size)
    valid_token = token_idx < seq_len
    valid = valid_token.view(1, n_blocks, block_size, 1)
    size = valid_token.sum(dim=1).long()
    size_float = size.clamp_min(1).float()

    # Accumulate summaries in FP32 without materializing full FP32 copies of
    # prompt K/V. Token pages stay in their model dtype for the Triton kernel.
    k_sum = (k_block_attn * valid).sum(dim=2, dtype=torch.float32)
    v_sum = (v_block_attn * valid).sum(dim=2, dtype=torch.float32)
    k_for_max = k_block_attn.masked_fill(~valid, float("-inf"))
    k_for_min = k_block_attn.masked_fill(~valid, float("inf"))
    v_norm = torch.linalg.vector_norm(v_block_attn, dim=-1, dtype=torch.float32)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    return {
        "k_block_attn": k_block_attn,
        "v_block_attn": v_block_attn,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_bar": k_sum / size_float.view(1, n_blocks, 1),
        "v_bar": v_sum / size_float.view(1, n_blocks, 1),
        "k_max": k_for_max.amax(dim=2).float(),
        "k_min": k_for_min.amin(dim=2).float(),
        "v_norm_max": v_norm.amax(dim=2),
        "v_norm_all": v_norm.amax(dim=2).amax(dim=-1),
        "block_valid_counts": size,
    }


def _select_prompt_blocks(q_grouped, prefix, eps):
    if q_grouped.is_cuda and os.environ.get("CONDITION_BLOCK_EAGER_SELECTION") != "1":
        return _select_prompt_blocks_triton(q_grouped, prefix, eps)

    if os.environ.get("CONDITION_BLOCK_COMPILE_SELECTION") == "1":
        return _select_prompt_blocks_compiled(
            q_grouped,
            prefix["block_valid_counts"],
            prefix["k_bar"],
            prefix["k_max"],
            prefix["k_min"],
            prefix["v_norm_max"],
            prefix["v_bar"],
            float(eps),
        )

    return _select_prompt_blocks_eager(q_grouped, prefix, eps)


def _select_prompt_blocks_eager(q_grouped, prefix, eps):
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


def _select_prompt_blocks_tensor(
    q_grouped,
    block_valid_counts,
    k_bar,
    k_max,
    k_min,
    v_norm_max,
    v_bar_prefix,
    eps,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    size = block_valid_counts.view(1, 1, -1).long()
    cluster_exists = size > 0
    cluster_exists_view = cluster_exists.view(1, 1, 1, -1)
    size_float = size.clamp_min(1).float()

    s_c = torch.einsum("gsqd,gbd->gsqb", q_grouped, k_bar) / scale
    q_bounds = q_grouped[:, :, :, None, :]
    q_k_max = q_bounds * k_max[:, None, None]
    q_k_min = q_bounds * k_min[:, None, None]
    upper = torch.maximum(q_k_max, q_k_min).sum(dim=-1) / scale
    lower = torch.minimum(q_k_max, q_k_min).sum(dim=-1) / scale
    delta = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
    delta = delta.masked_fill(~cluster_exists_view, 0.0)

    b_c = v_norm_max[:, None, None, :].expand(n_kv_heads, group_size, n_query, -1)
    b_c = b_c.masked_fill(~cluster_exists_view, 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).unsqueeze(2) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists_view, float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    cosh_delta = torch.cosh(delta)
    denom = (p_tensor * cosh_delta).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (cosh_delta - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )

    selected = (condition.mean(dim=1, keepdim=True) > eps) & cluster_exists_view
    selected = selected.expand(n_kv_heads, group_size, n_query, -1)
    v_bar = v_bar_prefix[:, None, None].expand(
        n_kv_heads,
        group_size,
        n_query,
        -1,
        head_dim,
    )
    return selected, z_logits, v_bar, size.view(1, -1), cluster_exists.view(1, -1)


_select_prompt_blocks_compiled = torch.compile(
    _select_prompt_blocks_tensor,
    mode="reduce-overhead",
    fullgraph=False,
)


@triton.jit
def _condition_block_selection_kernel(
    q_ptr,
    k_bar_ptr,
    k_max_ptr,
    k_min_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    selected_ptr,
    z_out_ptr,
    n_blocks: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_head = tl.program_id(0)
    g = tl.arange(0, BLOCK_G)
    b_lane = tl.arange(0, BLOCK_B)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    d_mask = d < head_dim
    row = kv_head * group_size + g
    q = tl.load(
        q_ptr + row[:, None] * head_dim + d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    z_m = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    z_l = tl.zeros((BLOCK_G,), tl.float32)
    c_m = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    c_l = tl.zeros((BLOCK_G,), tl.float32)

    # Pass 1: log-sum-exp for Z=sum(exp(z)) and
    # C=sum(exp(z)*cosh(delta)).
    b_start = 0
    while b_start < n_blocks:
        b = b_start + b_lane
        valid = b < n_blocks
        count = tl.load(counts_ptr + b, mask=valid, other=0)
        active = valid & (count > 0)
        stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
        k_bar = tl.load(
            k_bar_ptr + stat_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_hi = tl.load(
            k_max_ptr + stat_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_lo = tl.load(
            k_min_ptr + stat_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
        upper_k = tl.where(q[:, None, :] >= 0.0, k_hi[None, :, :], k_lo[None, :, :])
        lower_k = tl.where(q[:, None, :] >= 0.0, k_lo[None, :, :], k_hi[None, :, :])
        upper = tl.sum(q[:, None, :] * upper_k, axis=2) * scale
        lower = tl.sum(q[:, None, :] * lower_k, axis=2) * scale
        delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        z = tl.where(g_mask[:, None] & active[None, :], z, -float("inf"))
        cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
        zc = z + tl.log(cosh_delta)

        tile_z_m = tl.max(z, axis=1)
        new_z_m = tl.maximum(z_m, tile_z_m)
        z_l = z_l * tl.exp(z_m - new_z_m) + tl.sum(tl.exp(z - new_z_m[:, None]), axis=1)
        z_m = new_z_m
        tile_c_m = tl.max(zc, axis=1)
        new_c_m = tl.maximum(c_m, tile_c_m)
        c_l = c_l * tl.exp(c_m - new_c_m) + tl.sum(tl.exp(zc - new_c_m[:, None]), axis=1)
        c_m = new_c_m
        b_start += BLOCK_B

    b_all = tl.load(v_norm_all_ptr + kv_head).to(tl.float32)

    # Pass 2: recompute the inexpensive bounds, apply the condition, and emit
    # the compact routing state consumed by stage2.
    b_start = 0
    while b_start < n_blocks:
        b = b_start + b_lane
        valid = b < n_blocks
        count = tl.load(counts_ptr + b, mask=valid, other=0)
        active = valid & (count > 0)
        stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
        k_bar = tl.load(k_bar_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        k_hi = tl.load(k_max_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        k_lo = tl.load(k_min_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
        upper_k = tl.where(q[:, None, :] >= 0.0, k_hi[None, :, :], k_lo[None, :, :])
        lower_k = tl.where(q[:, None, :] >= 0.0, k_lo[None, :, :], k_hi[None, :, :])
        upper = tl.sum(q[:, None, :] * upper_k, axis=2) * scale
        lower = tl.sum(q[:, None, :] * lower_k, axis=2) * scale
        delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        z = tl.where(g_mask[:, None] & active[None, :], z, -float("inf"))
        cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
        tanh_half = 2.0 / (1.0 + tl.exp(-delta)) - 1.0
        b_c = tl.load(v_norm_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)
        term1 = 2.0 * b_all * tl.exp(z - c_m[:, None]) * (cosh_delta - 1.0) / c_l[:, None]
        term2 = 2.0 * b_c[None, :] * tl.exp(z - z_m[:, None]) * tanh_half / z_l[:, None]
        condition = tl.where(g_mask[:, None] & active[None, :], term1 + term2, 0.0)
        selected = (tl.sum(condition, axis=0) / group_size) > eps
        tl.store(
            selected_ptr + row[:, None] * n_blocks + b[None, :],
            selected[None, :],
            mask=g_mask[:, None] & valid[None, :],
        )
        tl.store(
            z_out_ptr + row[:, None] * n_blocks + b[None, :],
            z,
            mask=g_mask[:, None] & valid[None, :],
        )
        b_start += BLOCK_B


def _select_prompt_blocks_triton(q_grouped, prefix, eps):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        return _select_prompt_blocks_eager(q_grouped, prefix, eps)
    n_blocks = int(prefix["block_valid_counts"].numel())
    rows = int(n_kv_heads * group_size)
    q = q_grouped.reshape(rows, head_dim).contiguous()
    selected = torch.empty((rows, n_blocks), device=q.device, dtype=torch.bool)
    z_logits = torch.empty((rows, n_blocks), device=q.device, dtype=torch.float32)
    selection_chunk = 16
    n_chunks = triton.cdiv(n_blocks, selection_chunk)
    s_cache = torch.empty((rows, n_blocks), device=q.device, dtype=torch.float32)
    delta_cache = torch.empty_like(s_cache)
    partial = torch.empty((4, rows, n_chunks), device=q.device, dtype=torch.float32)
    _condition_block_selection_stats_kernel[(n_kv_heads, n_chunks)](
        q,
        prefix["k_bar"].contiguous(),
        prefix["k_max"].contiguous(),
        prefix["k_min"].contiguous(),
        prefix["block_valid_counts"].contiguous(),
        s_cache,
        delta_cache,
        partial[0],
        partial[1],
        partial[2],
        partial[3],
        n_blocks,
        n_chunks,
        group_size,
        head_dim,
        head_dim**-0.5,
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_B=selection_chunk,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    global_stats = torch.empty((4, rows), device=q.device, dtype=torch.float32)
    _condition_block_selection_reduce_kernel[(rows,)](
        partial[0],
        partial[1],
        partial[2],
        partial[3],
        global_stats[0],
        global_stats[1],
        global_stats[2],
        global_stats[3],
        n_chunks,
        BLOCK_C=triton.next_power_of_2(n_chunks),
        num_warps=4,
    )
    _condition_block_selection_finalize_kernel[(n_kv_heads, n_chunks)](
        s_cache,
        delta_cache,
        prefix["v_norm_max"].contiguous(),
        prefix["v_norm_all"].contiguous(),
        prefix["block_valid_counts"].contiguous(),
        global_stats[0],
        global_stats[1],
        global_stats[2],
        global_stats[3],
        selected,
        z_logits,
        n_blocks,
        n_chunks,
        group_size,
        float(eps),
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_B=selection_chunk,
        num_warps=4,
    )
    selected = selected.reshape(n_kv_heads, group_size, 1, n_blocks)
    z_logits = z_logits.reshape(n_kv_heads, group_size, 1, n_blocks)
    size = prefix["block_valid_counts"].view(1, -1)
    cluster_exists = size > 0
    v_bar = prefix["v_bar"][:, None, None].expand(
        n_kv_heads, group_size, 1, n_blocks, head_dim
    )
    return selected, z_logits, v_bar, size, cluster_exists


@triton.jit
def _condition_block_selection_stats_kernel(
    q_ptr,
    k_bar_ptr,
    k_max_ptr,
    k_min_ptr,
    counts_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    n_blocks: tl.constexpr,
    n_chunks: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    b_mask = b < n_blocks
    d_mask = d < head_dim
    row = kv_head * group_size + g
    q = tl.load(q_ptr + row[:, None] * head_dim + d[None, :], mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
    k_bar = tl.load(k_bar_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    k_hi = tl.load(k_max_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    k_lo = tl.load(k_min_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
    upper_k = tl.where(q[:, None, :] >= 0.0, k_hi[None, :, :], k_lo[None, :, :])
    lower_k = tl.where(q[:, None, :] >= 0.0, k_lo[None, :, :], k_hi[None, :, :])
    upper = tl.sum(q[:, None, :] * upper_k, axis=2) * scale
    lower = tl.sum(q[:, None, :] * lower_k, axis=2) * scale
    delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    active_2d = g_mask[:, None] & active[None, :]
    z = tl.where(active_2d, z, -float("inf"))
    cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
    zc = z + tl.log(cosh_delta)
    tl.store(s_cache_ptr + row[:, None] * n_blocks + b[None, :], s, mask=active_2d)
    tl.store(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], delta, mask=active_2d)
    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=g_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=g_mask)
    tl.store(c_m_ptr + partial_off, c_m, mask=g_mask)
    tl.store(c_l_ptr + partial_off, c_l, mask=g_mask)


@triton.jit
def _condition_block_selection_reduce_kernel(
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    global_z_m_ptr,
    global_z_l_ptr,
    global_c_m_ptr,
    global_c_l_ptr,
    n_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    mask = c < n_chunks
    z_m = tl.load(z_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    z_l = tl.load(z_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + row * n_chunks + c, mask=mask, other=-float("inf"))
    c_l = tl.load(c_l_ptr + row * n_chunks + c, mask=mask, other=0.0)
    global_z_m = tl.max(z_m, axis=0)
    global_c_m = tl.max(c_m, axis=0)
    global_z_l = tl.sum(z_l * tl.exp(z_m - global_z_m), axis=0)
    global_c_l = tl.sum(c_l * tl.exp(c_m - global_c_m), axis=0)
    tl.store(global_z_m_ptr + row, global_z_m)
    tl.store(global_z_l_ptr + row, global_z_l)
    tl.store(global_c_m_ptr + row, global_c_m)
    tl.store(global_c_l_ptr + row, global_c_l)


@triton.jit
def _condition_block_selection_finalize_kernel(
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    selected_ptr,
    z_out_ptr,
    n_blocks: tl.constexpr,
    n_chunks: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    g_mask = g < group_size
    b_mask = b < n_blocks
    row = kv_head * group_size + g
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    mask_2d = g_mask[:, None] & active[None, :]
    s = tl.load(s_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0)
    delta = tl.load(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], mask=mask_2d, other=0.0)
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    z_m = tl.load(z_m_ptr + row, mask=g_mask, other=0.0)
    z_l = tl.load(z_l_ptr + row, mask=g_mask, other=1.0)
    c_m = tl.load(c_m_ptr + row, mask=g_mask, other=0.0)
    c_l = tl.load(c_l_ptr + row, mask=g_mask, other=1.0)
    cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
    tanh_half = 2.0 / (1.0 + tl.exp(-delta)) - 1.0
    b_c = tl.load(v_norm_ptr + kv_head * n_blocks + b, mask=active, other=0.0)
    b_all = tl.load(v_norm_all_ptr + kv_head)
    term1 = 2.0 * b_all * tl.exp(z - c_m[:, None]) * (cosh_delta - 1.0) / c_l[:, None]
    term2 = 2.0 * b_c[None, :] * tl.exp(z - z_m[:, None]) * tanh_half / z_l[:, None]
    condition = tl.where(mask_2d, term1 + term2, 0.0)
    selected = (tl.sum(condition, axis=0) / group_size) > eps
    tl.store(selected_ptr + row[:, None] * n_blocks + b[None, :], selected[None, :], mask=g_mask[:, None] & b_mask[None, :])
    tl.store(z_out_ptr + row[:, None] * n_blocks + b[None, :], z, mask=g_mask[:, None] & b_mask[None, :])


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
    attention_dtype,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    selected, z_logits, v_bar, size, cluster_exists = _select_prompt_blocks(
        q_grouped,
        prompt_prefix,
        eps,
    )

    if os.environ.get("CONDITION_BLOCK_DENSE_STAGE2") == "1":
        output = _condition_block_decode_output_dense(
            q_grouped=q_grouped,
            pos_tensor=pos_tensor,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix.float(),
            v_suffix=v_suffix.float(),
            block_size=block_size,
            prompt_len=prompt_len,
            selected=selected,
            z_logits=z_logits,
            v_bar=v_bar,
            cluster_exists=cluster_exists,
        )
    elif os.environ.get("CONDITION_BLOCK_COMPACT_SDPA_STAGE2") == "1":
        output = _condition_block_decode_output_compact_sdpa(
            q_grouped=q_grouped,
            pos_tensor=pos_tensor,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix.float(),
            v_suffix=v_suffix.float(),
            block_size=block_size,
            prompt_len=prompt_len,
            selected=selected,
            attention_dtype=attention_dtype,
        )
    else:
        output = _condition_block_decode_output_triton(
            q_grouped=q_grouped,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            selected=selected,
            z_logits=z_logits,
            attention_dtype=attention_dtype,
        )

    stats = None
    if os.environ.get("CONDITION_BLOCK_SKIP_STATS") != "1":
        stats = _condition_stats(
            selected=selected,
            size=size,
            cluster_exists=cluster_exists,
            pos_tensor=pos_tensor,
            prompt_len=prompt_len,
        )
    return output, stats


@triton.jit
def _condition_block_stage2_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    k_bar_ptr,
    v_bar_ptr,
    selected_ptr,
    counts_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    out_ptr,
    n_blocks: tl.constexpr,
    suffix_len,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = row // group_size
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < head_dim

    q = tl.load(q_ptr + row * head_dim + offs, mask=d_mask, other=0.0).to(tl.float32)
    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    # Keep the scalar softmax state in element 0 and broadcast it when updating acc.
    m_scalar = tl.full((), -float("inf"), tl.float32)
    l_scalar = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for block_idx in tl.range(0, n_blocks):
        count = tl.load(counts_ptr + block_idx)
        selected = tl.load(selected_ptr + row * n_blocks + block_idx)
        if selected:
            for token_idx in tl.static_range(0, block_size):
                valid = token_idx < count
                k_off = (((kv_head * n_blocks + block_idx) * block_size + token_idx) * head_dim) + offs
                v_off = k_off
                k = tl.load(k_block_ptr + k_off, mask=d_mask & valid, other=0.0).to(tl.float32)
                v = tl.load(v_block_ptr + v_off, mask=d_mask & valid, other=0.0).to(tl.float32)
                score = tl.sum(q * k, axis=0) * scale
                score = tl.where(valid, score, -float("inf"))
                new_m = tl.maximum(m_scalar, score)
                alpha = tl.exp(m_scalar - new_m)
                beta = tl.exp(score - new_m)
                acc = acc * alpha + v * beta
                l_scalar = l_scalar * alpha + beta
                m_scalar = new_m
        else:
            valid = count > 0
            k_off = ((kv_head * n_blocks + block_idx) * head_dim) + offs
            v_off = k_off
            k = tl.load(k_bar_ptr + k_off, mask=d_mask & valid, other=0.0).to(tl.float32)
            v = tl.load(v_bar_ptr + v_off, mask=d_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * k, axis=0) * scale + tl.log(count.to(tl.float32))
            score = tl.where(valid, score, -float("inf"))
            new_m = tl.maximum(m_scalar, score)
            alpha = tl.exp(m_scalar - new_m)
            beta = tl.exp(score - new_m)
            acc = acc * alpha + v * beta
            l_scalar = l_scalar * alpha + beta
            m_scalar = new_m

    suffix_idx = 0
    while suffix_idx < suffix_len:
        k_off = ((kv_head * suffix_len + suffix_idx) * head_dim) + offs
        v_off = k_off
        k = tl.load(k_suffix_ptr + k_off, mask=d_mask, other=0.0).to(tl.float32)
        v = tl.load(v_suffix_ptr + v_off, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        new_m = tl.maximum(m_scalar, score)
        alpha = tl.exp(m_scalar - new_m)
        beta = tl.exp(score - new_m)
        acc = acc * alpha + v * beta
        l_scalar = l_scalar * alpha + beta
        m_scalar = new_m
        suffix_idx += 1

    out = acc / tl.maximum(l_scalar, 1.0e-30)
    tl.store(out_ptr + row * head_dim + offs, out, mask=d_mask)


def _condition_block_decode_output_triton(
    *,
    q_grouped,
    prompt_prefix,
    k_suffix,
    v_suffix,
    selected,
    z_logits,
    attention_dtype,
):
    if q_grouped.shape[2] != 1:
        raise ValueError("triton condition_block stage2 expects q_len=1.")
    if not q_grouped.is_cuda:
        return _condition_block_decode_output_compact_sdpa(
            q_grouped=q_grouped,
            pos_tensor=torch.zeros((1,), device=q_grouped.device, dtype=torch.long),
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            block_size=int(prompt_prefix["valid_token"].shape[1]),
            prompt_len=0,
            selected=selected,
            attention_dtype=attention_dtype,
        )

    if os.environ.get("CONDITION_BLOCK_TRITON_ROW_STAGE2") == "1":
        return _condition_block_decode_output_triton_row(
            q_grouped=q_grouped,
            prompt_prefix=prompt_prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            selected=selected,
        )

    n_kv_heads, group_size, _n_query, head_dim = q_grouped.shape
    n_blocks = int(selected.shape[-1])
    suffix_len = int(k_suffix.shape[1])
    rows = int(n_kv_heads * group_size)
    block_size = int(prompt_prefix["valid_token"].shape[1])
    block_d = triton.next_power_of_2(head_dim)
    chunk_blocks = int(os.environ.get("CONDITION_BLOCK_TRITON_CHUNK_BLOCKS", "16"))
    n_chunks = triton.cdiv(n_blocks, chunk_blocks)

    q = q_grouped.reshape(rows, head_dim).contiguous()
    selected_rows = selected[:, :, 0, :].contiguous().reshape(rows, n_blocks)
    z_rows = z_logits[:, :, 0, :].contiguous().reshape(rows, n_blocks)
    partial_acc = torch.empty((rows, n_chunks, head_dim), device=q.device, dtype=torch.float32)
    partial_m = torch.empty((rows, n_chunks), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((rows, n_chunks), device=q.device, dtype=torch.float32)

    common_args = (
        q,
        prompt_prefix["k_block_attn"].contiguous(),
        prompt_prefix["v_block_attn"].contiguous(),
        prompt_prefix["k_bar"].contiguous(),
        prompt_prefix["v_bar"].contiguous(),
        selected_rows,
        prompt_prefix["block_valid_counts"].contiguous(),
        k_suffix.contiguous(),
        v_suffix.contiguous(),
        partial_acc,
        partial_m,
        partial_l,
        n_blocks,
        suffix_len,
        group_size,
        head_dim,
        block_size,
        n_chunks,
        chunk_blocks,
        head_dim**-0.5,
    )
    use_per_head_kernel = (
        os.environ.get("CONDITION_BLOCK_TRITON_PER_HEAD_STAGE2") == "1"
        or block_size != 16
    )
    use_vector_gqa_kernel = os.environ.get("CONDITION_BLOCK_TRITON_VECTOR_GQA_STAGE2") == "1"
    if use_per_head_kernel:
        _condition_block_stage2_chunk_kernel[(rows, n_chunks)](
            *common_args,
            BLOCK_D=block_d,
            REP_TILE=16,
            SELECTED_BLOCK_TILE=2,
            SUFFIX_TILE=16,
            num_warps=4,
        )
    elif use_vector_gqa_kernel:
        _condition_block_stage2_gqa_chunk_kernel[(n_kv_heads, n_chunks)](
            *common_args,
            BLOCK_G=triton.next_power_of_2(group_size),
            BLOCK_D=block_d,
            REP_TILE=16,
            SELECTED_BLOCK_TILE=1,
            SUFFIX_TILE=16,
            num_warps=4,
        )
    else:
        _condition_block_stage2_tensorcore_kernel[(n_kv_heads, n_chunks)](
            q,
            prompt_prefix["k_block_attn"].contiguous(),
            prompt_prefix["v_block_attn"].contiguous(),
            prompt_prefix["v_bar"].contiguous(),
            selected_rows,
            z_rows,
            prompt_prefix["block_valid_counts"].contiguous(),
            k_suffix.contiguous(),
            v_suffix.contiguous(),
            partial_acc,
            partial_m,
            partial_l,
            n_blocks,
            suffix_len,
            group_size,
            head_dim,
            block_size,
            n_chunks,
            chunk_blocks,
            head_dim**-0.5,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_D=block_d,
            num_warps=4,
        )
    out = torch.empty((rows, head_dim), device=q.device, dtype=torch.float32)
    block_c = triton.next_power_of_2(n_chunks)
    _condition_block_stage2_reduce_kernel[(rows,)](
        partial_acc,
        partial_m,
        partial_l,
        out,
        n_chunks,
        head_dim,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out.reshape(n_kv_heads, group_size, 1, head_dim)


def _condition_block_decode_output_triton_row(
    *,
    q_grouped,
    prompt_prefix,
    k_suffix,
    v_suffix,
    selected,
):
    n_kv_heads, group_size, _n_query, head_dim = q_grouped.shape
    n_blocks = int(selected.shape[-1])
    suffix_len = int(k_suffix.shape[1])
    rows = int(n_kv_heads * group_size)
    block_size = int(prompt_prefix["valid_token"].shape[1])
    block_d = triton.next_power_of_2(head_dim)

    q = q_grouped.reshape(rows, head_dim).contiguous()
    out = torch.empty((rows, head_dim), device=q.device, dtype=torch.float32)
    selected_rows = selected[:, :, 0, :].contiguous().reshape(rows, n_blocks)

    _condition_block_stage2_kernel[(rows,)](
        q,
        prompt_prefix["k_block_attn"].contiguous(),
        prompt_prefix["v_block_attn"].contiguous(),
        prompt_prefix["k_bar"].contiguous(),
        prompt_prefix["v_bar"].contiguous(),
        selected_rows,
        prompt_prefix["block_valid_counts"].contiguous(),
        k_suffix.contiguous(),
        v_suffix.contiguous(),
        out,
        n_blocks,
        suffix_len,
        group_size,
        head_dim,
        block_size,
        head_dim**-0.5,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out.reshape(n_kv_heads, group_size, 1, head_dim)


@triton.jit
def _condition_block_stage2_chunk_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    k_bar_ptr,
    v_bar_ptr,
    selected_ptr,
    counts_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks: tl.constexpr,
    suffix_len,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    n_chunks: tl.constexpr,
    chunk_blocks: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    REP_TILE: tl.constexpr,
    SELECTED_BLOCK_TILE: tl.constexpr,
    SUFFIX_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    kv_head = row // group_size
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < head_dim

    q = tl.load(q_ptr + row * head_dim + offs, mask=d_mask, other=0.0).to(tl.float32)
    m_scalar = tl.full((), -float("inf"), tl.float32)
    l_scalar = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    block_start = chunk * chunk_blocks
    block_end = tl.minimum(block_start + chunk_blocks, n_blocks)

    # Stream 1: unexpanded blocks. Process several representatives together so
    # q @ k_bar and the value reduction are vectorized instead of issuing one
    # scalar dot product per block.
    rep_start = block_start
    rep_lane = tl.arange(0, REP_TILE)
    while rep_start < block_end:
        block_idx = rep_start + rep_lane
        valid_block = block_idx < block_end
        count = tl.load(counts_ptr + block_idx, mask=valid_block, other=0)
        is_selected = tl.load(
            selected_ptr + row * n_blocks + block_idx,
            mask=valid_block,
            other=1,
        ).to(tl.int1)
        active_rep = valid_block & (count > 0) & (~is_selected)
        k_off = ((kv_head * n_blocks + block_idx[:, None]) * head_dim) + offs[None, :]
        k_rep = tl.load(
            k_bar_ptr + k_off,
            mask=active_rep[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rep_scores = tl.sum(k_rep * q[None, :], axis=1) * scale
        rep_scores += tl.log(tl.maximum(count, 1).to(tl.float32))
        rep_scores = tl.where(active_rep, rep_scores, -float("inf"))

        has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
        rep_m = tl.max(rep_scores, axis=0)
        new_m = tl.where(has_rep, tl.maximum(m_scalar, rep_m), m_scalar)
        alpha = tl.where(has_rep, tl.exp(m_scalar - new_m), 1.0)
        beta = tl.where(active_rep, tl.exp(rep_scores - new_m), 0.0)
        v_rep = tl.load(
            v_bar_ptr + k_off,
            mask=active_rep[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(v_rep * beta[:, None], axis=0)
        l_scalar = l_scalar * alpha + tl.sum(beta, axis=0)
        m_scalar = new_m
        rep_start += REP_TILE

    # Stream 2: expanded blocks. Masked loads ensure that token K/V are read
    # only for selected blocks; unselected pages never enter SRAM. Multiple
    # selected pages are handled as one token tile to expose parallelism.
    selected_start = block_start
    token_lane = tl.arange(0, SELECTED_BLOCK_TILE * block_size)
    while selected_start < block_end:
        local_block = token_lane // block_size
        token_idx = token_lane - local_block * block_size
        block_idx = selected_start + local_block
        valid_block = block_idx < block_end
        count = tl.load(counts_ptr + block_idx, mask=valid_block, other=0)
        is_selected = tl.load(
            selected_ptr + row * n_blocks + block_idx,
            mask=valid_block,
            other=0,
        ).to(tl.int1)
        active_token = valid_block & is_selected & (token_idx < count)
        token_base = (
            ((kv_head * n_blocks + block_idx[:, None]) * block_size + token_idx[:, None])
            * head_dim
        )
        k_token = tl.load(
            k_block_ptr + token_base + offs[None, :],
            mask=active_token[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        token_scores = tl.sum(k_token * q[None, :], axis=1) * scale
        token_scores = tl.where(active_token, token_scores, -float("inf"))

        has_token = tl.sum(active_token.to(tl.int32), axis=0) > 0
        token_m = tl.max(token_scores, axis=0)
        new_m = tl.where(has_token, tl.maximum(m_scalar, token_m), m_scalar)
        alpha = tl.where(has_token, tl.exp(m_scalar - new_m), 1.0)
        beta = tl.where(active_token, tl.exp(token_scores - new_m), 0.0)
        v_token = tl.load(
            v_block_ptr + token_base + offs[None, :],
            mask=active_token[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(v_token * beta[:, None], axis=0)
        l_scalar = l_scalar * alpha + tl.sum(beta, axis=0)
        m_scalar = new_m
        selected_start += SELECTED_BLOCK_TILE

    if chunk == n_chunks - 1:
        suffix_start = 0
        suffix_lane = tl.arange(0, SUFFIX_TILE)
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + suffix_lane
            active_suffix = suffix_idx < suffix_len
            suffix_off = ((kv_head * suffix_len + suffix_idx[:, None]) * head_dim) + offs[None, :]
            k_suffix = tl.load(
                k_suffix_ptr + suffix_off,
                mask=active_suffix[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            suffix_scores = tl.sum(k_suffix * q[None, :], axis=1) * scale
            suffix_scores = tl.where(active_suffix, suffix_scores, -float("inf"))
            suffix_m = tl.max(suffix_scores, axis=0)
            new_m = tl.maximum(m_scalar, suffix_m)
            alpha = tl.exp(m_scalar - new_m)
            beta = tl.where(active_suffix, tl.exp(suffix_scores - new_m), 0.0)
            v_suffix = tl.load(
                v_suffix_ptr + suffix_off,
                mask=active_suffix[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(v_suffix * beta[:, None], axis=0)
            l_scalar = l_scalar * alpha + tl.sum(beta, axis=0)
            m_scalar = new_m
            suffix_start += SUFFIX_TILE

    base = (row * n_chunks + chunk)
    tl.store(partial_m_ptr + base, m_scalar)
    tl.store(partial_l_ptr + base, l_scalar)
    tl.store(partial_acc_ptr + base * head_dim + offs, acc, mask=d_mask)


@triton.jit
def _condition_block_stage2_gqa_chunk_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    k_bar_ptr,
    v_bar_ptr,
    selected_ptr,
    counts_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks: tl.constexpr,
    suffix_len,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    n_chunks: tl.constexpr,
    chunk_blocks: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    REP_TILE: tl.constexpr,
    SELECTED_BLOCK_TILE: tl.constexpr,
    SUFFIX_TILE: tl.constexpr,
):
    """Partitioned hybrid attention with K/V reuse across a GQA group."""
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    d_mask = d < head_dim
    row = kv_head * group_size + g

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    m = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_G,), tl.float32)
    acc = tl.zeros((BLOCK_G, BLOCK_D), tl.float32)
    block_start = chunk * chunk_blocks
    block_end = tl.minimum(block_start + chunk_blocks, n_blocks)

    # Representatives: K/V are loaded once and reused by every query head that
    # shares this KV head.
    rep_lane = tl.arange(0, REP_TILE)
    rep_start = block_start
    while rep_start < block_end:
        block_idx = rep_start + rep_lane
        valid_block = block_idx < block_end
        count = tl.load(counts_ptr + block_idx, mask=valid_block, other=0)
        # Selection is shared within a GQA group by construction.
        is_selected = tl.load(
            selected_ptr + (kv_head * group_size) * n_blocks + block_idx,
            mask=valid_block,
            other=1,
        ).to(tl.int1)
        active = valid_block & (count > 0) & (~is_selected)
        kv_off = ((kv_head * n_blocks + block_idx[:, None]) * head_dim) + d[None, :]
        k = tl.load(
            k_bar_ptr + kv_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        scores += tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        scores = tl.where(g_mask[:, None] & active[None, :], scores, -float("inf"))
        has_value = tl.sum(active.to(tl.int32), axis=0) > 0
        tile_m = tl.max(scores, axis=1)
        new_m = tl.where(has_value & g_mask, tl.maximum(m, tile_m), m)
        alpha = tl.where(has_value & g_mask, tl.exp(m - new_m), 1.0)
        beta = tl.where(active[None, :] & g_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
        v = tl.load(
            v_bar_ptr + kv_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(beta[:, :, None] * v[None, :, :], axis=1)
        l = l * alpha + tl.sum(beta, axis=1)
        m = new_m
        rep_start += REP_TILE

    # Selected pages: masked loads are the only accesses to token-level prompt
    # K/V, and each loaded page is reused by the whole GQA group.
    token_lane = tl.arange(0, SELECTED_BLOCK_TILE * block_size)
    selected_start = block_start
    while selected_start < block_end:
        local_block = token_lane // block_size
        token_idx = token_lane - local_block * block_size
        block_idx = selected_start + local_block
        valid_block = block_idx < block_end
        count = tl.load(counts_ptr + block_idx, mask=valid_block, other=0)
        is_selected = tl.load(
            selected_ptr + (kv_head * group_size) * n_blocks + block_idx,
            mask=valid_block,
            other=0,
        ).to(tl.int1)
        active = valid_block & is_selected & (token_idx < count)
        token_off = (
            ((kv_head * n_blocks + block_idx[:, None]) * block_size + token_idx[:, None])
            * head_dim
            + d[None, :]
        )
        k = tl.load(
            k_block_ptr + token_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        scores = tl.where(g_mask[:, None] & active[None, :], scores, -float("inf"))
        has_value = tl.sum(active.to(tl.int32), axis=0) > 0
        tile_m = tl.max(scores, axis=1)
        new_m = tl.where(has_value & g_mask, tl.maximum(m, tile_m), m)
        alpha = tl.where(has_value & g_mask, tl.exp(m - new_m), 1.0)
        beta = tl.where(active[None, :] & g_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
        v = tl.load(
            v_block_ptr + token_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(beta[:, :, None] * v[None, :, :], axis=1)
        l = l * alpha + tl.sum(beta, axis=1)
        m = new_m
        selected_start += SELECTED_BLOCK_TILE

    if chunk == n_chunks - 1:
        suffix_lane = tl.arange(0, SUFFIX_TILE)
        suffix_start = 0
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + suffix_lane
            active = suffix_idx < suffix_len
            suffix_off = ((kv_head * suffix_len + suffix_idx[:, None]) * head_dim) + d[None, :]
            k = tl.load(
                k_suffix_ptr + suffix_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(g_mask[:, None] & active[None, :], scores, -float("inf"))
            tile_m = tl.max(scores, axis=1)
            new_m = tl.where(g_mask, tl.maximum(m, tile_m), m)
            alpha = tl.where(g_mask, tl.exp(m - new_m), 1.0)
            beta = tl.where(active[None, :] & g_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
            v = tl.load(
                v_suffix_ptr + suffix_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(beta[:, :, None] * v[None, :, :], axis=1)
            l = l * alpha + tl.sum(beta, axis=1)
            m = new_m
            suffix_start += SUFFIX_TILE

    base = row * n_chunks + chunk
    tl.store(partial_m_ptr + base, m, mask=g_mask)
    tl.store(partial_l_ptr + base, l, mask=g_mask)
    tl.store(
        partial_acc_ptr + base[:, None] * head_dim + d[None, :],
        acc,
        mask=g_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _condition_block_stage2_tensorcore_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    v_bar_ptr,
    selected_ptr,
    z_logits_ptr,
    counts_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks: tl.constexpr,
    suffix_len,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    n_chunks: tl.constexpr,
    chunk_blocks: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Tensor-Core hybrid attention for one KV head and one partition.

    GQA query rows are padded to BLOCK_M. Each representative tile and each
    selected page form a BLOCK_M x BLOCK_N MMA, matching the 16-token page
    layout used by the KV cache.
    """
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    m_mask = m_off < group_size
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off
    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    block_start = chunk * chunk_blocks
    block_end = tl.minimum(block_start + chunk_blocks, n_blocks)

    # Dense representative tiles, with selected representatives masked out.
    rep_start = block_start
    while rep_start < block_end:
        block_idx = rep_start + n_off
        valid_block = block_idx < block_end
        count = tl.load(counts_ptr + block_idx, mask=valid_block, other=0)
        is_selected = tl.load(
            selected_ptr + (kv_head * group_size) * n_blocks + block_idx,
            mask=valid_block,
            other=1,
        ).to(tl.int1)
        active = valid_block & (count > 0) & (~is_selected)
        kv_off = ((kv_head * n_blocks + block_idx[:, None]) * head_dim) + d_off[None, :]
        scores = tl.load(
            z_logits_ptr + row[:, None] * n_blocks + block_idx[None, :],
            mask=m_mask[:, None] & active[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        scores = tl.where(m_mask[:, None] & active[None, :], scores, -float("inf"))
        has_value = tl.sum(active.to(tl.int32), axis=0) > 0
        tile_m = tl.max(scores, axis=1)
        new_m = tl.where(has_value & m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
        alpha = tl.where(has_value & m_mask, tl.exp(softmax_m - new_m), 1.0)
        beta = tl.where(active[None, :] & m_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
        v = tl.load(
            v_bar_ptr + kv_off,
            mask=active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v, out_dtype=tl.float32)
        softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
        softmax_m = new_m
        rep_start += BLOCK_N

    # Sparse page stream. The branch prevents any token-level K/V load for an
    # unselected block; a selected 16-token page is consumed as one MMA tile.
    block_idx = block_start
    while block_idx < block_end:
        count = tl.load(counts_ptr + block_idx)
        is_selected = tl.load(
            selected_ptr + (kv_head * group_size) * n_blocks + block_idx
        )
        if is_selected:
            active = n_off < count
            token_off = (
                ((kv_head * n_blocks + block_idx) * block_size + n_off[:, None])
                * head_dim
                + d_off[None, :]
            )
            k = tl.load(
                k_block_ptr + token_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(m_mask[:, None] & active[None, :], scores, -float("inf"))
            tile_m = tl.max(scores, axis=1)
            new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
            alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
            beta = tl.where(active[None, :] & m_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
            v = tl.load(
                v_block_ptr + token_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v, out_dtype=tl.float32)
            softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
            softmax_m = new_m
        block_idx += 1

    if chunk == n_chunks - 1:
        suffix_start = 0
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + n_off
            active = suffix_idx < suffix_len
            suffix_off = ((kv_head * suffix_len + suffix_idx[:, None]) * head_dim) + d_off[None, :]
            k = tl.load(
                k_suffix_ptr + suffix_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(m_mask[:, None] & active[None, :], scores, -float("inf"))
            tile_m = tl.max(scores, axis=1)
            new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
            alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
            beta = tl.where(active[None, :] & m_mask[:, None], tl.exp(scores - new_m[:, None]), 0.0)
            v = tl.load(
                v_suffix_ptr + suffix_off,
                mask=active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v, out_dtype=tl.float32)
            softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
            softmax_m = new_m
            suffix_start += BLOCK_N

    base = row * n_chunks + chunk
    tl.store(partial_m_ptr + base, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + base, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + base[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _condition_block_stage2_reduce_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    n_chunks: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    d = tl.arange(0, BLOCK_D)
    c_mask = c < n_chunks
    d_mask = d < head_dim

    m = tl.load(partial_m_ptr + row * n_chunks + c, mask=c_mask, other=-float("inf")).to(tl.float32)
    l = tl.load(partial_l_ptr + row * n_chunks + c, mask=c_mask, other=0.0).to(tl.float32)
    global_m = tl.max(m, axis=0)
    weights = tl.exp(m - global_m) * l
    denom = tl.sum(weights, axis=0)

    acc = tl.load(
        partial_acc_ptr + (row * n_chunks + c[:, None]) * head_dim + d[None, :],
        mask=c_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(acc * tl.exp(m[:, None] - global_m), axis=0)
    out = numerator / tl.maximum(denom, 1.0e-30)
    tl.store(out_ptr + row * head_dim + d, out, mask=d_mask)


def _condition_block_decode_output_dense(
    *,
    q_grouped,
    pos_tensor,
    prompt_prefix,
    k_suffix,
    v_suffix,
    block_size,
    prompt_len,
    selected,
    z_logits,
    v_bar,
    cluster_exists,
):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    visible = (
        prompt_prefix["valid_token"].view(1, 1, 1, -1, block_size)
        & (
            prompt_prefix["token_idx"].view(1, 1, 1, -1, block_size)
            <= pos_tensor.view(1, 1, -1, 1, 1)
        )
    )
    token_active = selected.unsqueeze(-1) & visible
    k_block = prompt_prefix["k_block_attn"].float()
    v_block = prompt_prefix["v_block_attn"].float()
    token_logits = torch.einsum("gsqd,gbtd->gsqbt", q_grouped, k_block) / scale
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
    numerator = torch.einsum("gsqbt,gbtd->gsqd", token_exp, v_block)
    numerator = numerator + (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=3)

    if suffix_logits is not None:
        suffix_exp = torch.exp(suffix_logits - max_logit[:, :, :, None]).masked_fill(
            ~suffix_active,
            0.0,
        )
        normalizer = normalizer + suffix_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("grqt,gtd->grqd", suffix_exp, v_suffix)

    output = numerator / normalizer.clamp_min(1e-30).unsqueeze(-1)
    return output


def _condition_block_decode_output_compact_sdpa(
    *,
    q_grouped,
    pos_tensor,
    prompt_prefix,
    k_suffix,
    v_suffix,
    block_size,
    prompt_len,
    selected,
    attention_dtype,
):
    if q_grouped.shape[2] != 1:
        raise ValueError("compact condition_block stage2 expects q_len=1.")

    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    n_blocks = int(selected.shape[-1])
    suffix_len = int(k_suffix.shape[1])
    rows = int(n_kv_heads * group_size * n_query)
    device = q_grouped.device

    selected = selected[:, :, 0, :]
    valid_token = prompt_prefix["valid_token"].view(1, 1, n_blocks, block_size)
    token_active = (selected.unsqueeze(-1) & valid_token).reshape(
        n_kv_heads,
        group_size,
        n_blocks * block_size,
    )
    cluster_active = (~selected) & prompt_prefix["block_valid_counts"].view(1, 1, n_blocks).gt(0)

    active_parts = [token_active, cluster_active]
    if suffix_len > 0:
        suffix_pos = torch.arange(
            int(prompt_len),
            int(prompt_len) + suffix_len,
            device=device,
            dtype=torch.long,
        )
        suffix_active = suffix_pos.view(1, 1, -1) <= pos_tensor.view(1, 1, 1)
        active_parts.append(suffix_active.expand(n_kv_heads, group_size, -1))
    active = torch.cat(active_parts, dim=-1).reshape(rows, -1)

    counts = active.sum(dim=-1)
    max_len = int(counts.max().item())
    if max_len <= 0:
        raise ValueError("condition_block compact attention received no active KV entries.")

    token_k = prompt_prefix["k_block_attn"].reshape(n_kv_heads, n_blocks * block_size, head_dim)
    token_v = prompt_prefix["v_block_attn"].reshape(n_kv_heads, n_blocks * block_size, head_dim)
    aug_k_parts = [token_k, prompt_prefix["k_bar"].to(dtype=token_k.dtype)]
    aug_v_parts = [token_v, prompt_prefix["v_bar"].to(dtype=token_v.dtype)]
    if suffix_len > 0:
        aug_k_parts.append(k_suffix.to(dtype=token_k.dtype))
        aug_v_parts.append(v_suffix.to(dtype=token_v.dtype))
    aug_k = torch.cat(aug_k_parts, dim=1).to(dtype=attention_dtype)
    aug_v = torch.cat(aug_v_parts, dim=1).to(dtype=attention_dtype)
    aug_k = aug_k[:, None].expand(n_kv_heads, group_size, -1, -1).reshape(rows, -1, head_dim)
    aug_v = aug_v[:, None].expand(n_kv_heads, group_size, -1, -1).reshape(rows, -1, head_dim)

    token_bias = torch.zeros((n_kv_heads, group_size, n_blocks * block_size), device=device)
    cluster_bias = torch.log(
        prompt_prefix["block_valid_counts"].clamp_min(1).float()
    ).view(1, 1, n_blocks).expand(n_kv_heads, group_size, -1)
    bias_parts = [token_bias, cluster_bias]
    if suffix_len > 0:
        bias_parts.append(torch.zeros((n_kv_heads, group_size, suffix_len), device=device))
    aug_bias = torch.cat(bias_parts, dim=-1).reshape(rows, -1)

    row_idx, src_idx = active.nonzero(as_tuple=True)
    dst_idx = (active.cumsum(dim=-1) - 1)[row_idx, src_idx]
    compact_k = torch.zeros((rows, max_len, head_dim), device=device, dtype=attention_dtype)
    compact_v = torch.zeros((rows, max_len, head_dim), device=device, dtype=attention_dtype)
    compact_bias = torch.full(
        (rows, max_len),
        torch.finfo(attention_dtype).min,
        device=device,
        dtype=attention_dtype,
    )
    compact_k[row_idx, dst_idx] = aug_k[row_idx, src_idx]
    compact_v[row_idx, dst_idx] = aug_v[row_idx, src_idx]
    compact_bias[row_idx, dst_idx] = aug_bias[row_idx, src_idx].to(dtype=attention_dtype)

    q = q_grouped.reshape(rows, n_query, head_dim).to(dtype=attention_dtype).unsqueeze(0)
    k = compact_k.unsqueeze(0)
    v = compact_v.unsqueeze(0)
    attn_mask = compact_bias.view(1, rows, 1, max_len)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=head_dim**-0.5,
    )
    return out.squeeze(0).reshape(n_kv_heads, group_size, n_query, head_dim).float()


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
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=float(dropout) if module.training else 0.0,
        scaling=scaling,
        **_kwargs,
    )


def _sdpa_prefill_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **_kwargs,
):
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(dropout) if module.training else 0.0,
        is_causal=True,
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
        self.skip_stats = os.environ.get("CONDITION_BLOCK_SKIP_STATS") == "1"

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

        # The persistent HF generate loop enters here for both prompt prefill
        # and decode. Only single-token decode is sparse; position follows the
        # current KV-cache length and therefore needs no Python step runner.
        if query.shape[2] != 1:
            return _sdpa_prefill_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
        self.pos = int(key.shape[2]) - 1

        if layer_idx is None or not self.should_compress(layer_idx):
            if layer_idx in self.layer_idx_set and not self.skip_stats:
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
            attention_dtype=query.dtype,
        )
        output = output.reshape(n_heads, q_len, head_dim).permute(1, 0, 2).unsqueeze(0)

        if stats is not None:
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


@contextlib.contextmanager
def model_attention_implementation(model, implementation):
    original = model.config._attn_implementation
    model.config._attn_implementation = implementation
    try:
        yield
    finally:
        model.config._attn_implementation = original


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
    if os.environ.get("CONDITION_BLOCK_SKIP_STATS") == "1":
        return {"condition_block_stats_disabled": True}

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
