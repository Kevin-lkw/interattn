"""Fallback and comparison implementations for condition-block stage2."""

import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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


@triton.jit(
    do_not_specialize=["n_chunks"],
    do_not_specialize_on_alignment=["n_chunks"],
)
def _condition_block_stage2_reduce_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    n_chunks,
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
