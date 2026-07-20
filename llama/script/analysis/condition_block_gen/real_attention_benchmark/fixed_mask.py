"""Optimized hybrid attention that consumes an externally captured page mask."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..methods.condition_block_triton_impl.legacy import (
    _condition_block_stage2_reduce_kernel,
)


@triton.jit(
    do_not_specialize=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
    do_not_specialize_on_alignment=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
)
def _fixed_mask_hybrid_attention_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    k_bar_ptr,
    v_bar_ptr,
    counts_ptr,
    selected_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    suffix_len,
    k_block_head_stride,
    v_block_head_stride,
    k_suffix_head_stride,
    k_suffix_token_stride,
    v_suffix_head_stride,
    v_suffix_token_stride,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    n_chunks,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Attend to representatives and only the externally selected token pages."""
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    p_off = tl.arange(0, PAGE_SIZE)
    m_mask = m_off < group_size
    block_idx = chunk * BLOCK_N + n_off
    block_mask = block_idx < n_blocks
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    count = tl.load(counts_ptr + block_idx, mask=block_mask, other=0)
    active_block = block_mask & (count > 0)
    selected = tl.load(
        selected_ptr + kv_head * n_blocks + block_idx,
        mask=block_mask,
        other=0,
    ).to(tl.int1) & active_block

    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Representatives remain real attention candidates. Their QK score must
    # still be computed even though routing itself is supplied externally.
    active_rep = active_block & (~selected)
    summary_off = (
        (kv_head * n_blocks + block_idx[:, None]) * head_dim + d_off[None, :]
    )
    k_rep = tl.load(
        k_bar_ptr + summary_off,
        mask=active_rep[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    rep_scores = tl.dot(q, tl.trans(k_rep), out_dtype=tl.float32) * scale
    rep_scores += tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    rep_scores = tl.where(
        m_mask[:, None] & active_rep[None, :], rep_scores, -float("inf")
    )
    has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
    rep_m = tl.max(rep_scores, axis=1)
    new_m = tl.where(has_rep & m_mask, rep_m, softmax_m)
    alpha = tl.where(has_rep & m_mask, tl.exp(softmax_m - new_m), 1.0)
    beta = tl.where(
        m_mask[:, None] & active_rep[None, :],
        tl.exp(rep_scores - new_m[:, None]),
        0.0,
    )
    v_rep = tl.load(
        v_bar_ptr + summary_off,
        mask=active_rep[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    acc = acc * alpha[:, None] + tl.dot(
        beta.to(tl.bfloat16), v_rep, out_dtype=tl.float32
    )
    softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
    softmax_m = new_m

    # Match the production page kernel: enumerate only selected pages and
    # consume an entire 16/32/64-token page in one MMA tile.
    sel_cum = tl.cumsum(selected.to(tl.int32), axis=0)
    n_sel = tl.sum(selected.to(tl.int32), axis=0)
    sel_j = 0
    while sel_j < n_sel:
        page_local = tl.sum(
            tl.where((sel_cum == sel_j + 1) & selected, n_off, 0), axis=0
        )
        page = chunk * BLOCK_N + page_local
        page_count = tl.load(counts_ptr + page)
        token_active = p_off < page_count
        token_off = (
            (page * PAGE_SIZE + p_off[:, None]) * head_dim + d_off[None, :]
        )
        page_k = tl.load(
            k_block_ptr + kv_head * k_block_head_stride + token_off,
            mask=token_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        page_scores = tl.dot(q, tl.trans(page_k), out_dtype=tl.float32) * scale
        page_scores = tl.where(
            m_mask[:, None] & token_active[None, :],
            page_scores,
            -float("inf"),
        )
        page_m = tl.max(page_scores, axis=1)
        page_new_m = tl.where(m_mask, tl.maximum(softmax_m, page_m), softmax_m)
        page_alpha = tl.where(m_mask, tl.exp(softmax_m - page_new_m), 1.0)
        page_beta = tl.where(
            m_mask[:, None] & token_active[None, :],
            tl.exp(page_scores - page_new_m[:, None]),
            0.0,
        )
        page_v = tl.load(
            v_block_ptr + kv_head * v_block_head_stride + token_off,
            mask=token_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc = acc * page_alpha[:, None] + tl.dot(
            page_beta.to(tl.bfloat16), page_v, out_dtype=tl.float32
        )
        softmax_l = softmax_l * page_alpha + tl.sum(page_beta, axis=1)
        softmax_m = page_new_m
        sel_j += 1

    if chunk == n_chunks - 1:
        suffix_start = 0
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + n_off
            suffix_active = suffix_idx < suffix_len
            k_suffix_off = (
                kv_head * k_suffix_head_stride
                + suffix_idx[:, None] * k_suffix_token_stride
                + d_off[None, :]
            )
            suffix_k = tl.load(
                k_suffix_ptr + k_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            suffix_scores = (
                tl.dot(q, tl.trans(suffix_k), out_dtype=tl.float32) * scale
            )
            suffix_scores = tl.where(
                m_mask[:, None] & suffix_active[None, :],
                suffix_scores,
                -float("inf"),
            )
            suffix_m = tl.max(suffix_scores, axis=1)
            suffix_new_m = tl.where(
                m_mask, tl.maximum(softmax_m, suffix_m), softmax_m
            )
            suffix_alpha = tl.where(
                m_mask, tl.exp(softmax_m - suffix_new_m), 1.0
            )
            suffix_beta = tl.where(
                m_mask[:, None] & suffix_active[None, :],
                tl.exp(suffix_scores - suffix_new_m[:, None]),
                0.0,
            )
            v_suffix_off = (
                kv_head * v_suffix_head_stride
                + suffix_idx[:, None] * v_suffix_token_stride
                + d_off[None, :]
            )
            suffix_v = tl.load(
                v_suffix_ptr + v_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * suffix_alpha[:, None] + tl.dot(
                suffix_beta.to(tl.bfloat16), suffix_v, out_dtype=tl.float32
            )
            softmax_l = softmax_l * suffix_alpha + tl.sum(suffix_beta, axis=1)
            softmax_m = suffix_new_m
            suffix_start += BLOCK_N

    partial_off = row * n_chunks + chunk
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


def _workspace_tensor(workspace, name, shape, *, device, dtype):
    tensor = workspace.get(name)
    if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        workspace[name] = tensor
    return tensor


def fixed_mask_hybrid_attention(
    *,
    q_grouped,
    prompt_prefix,
    selected,
    k_suffix,
    v_suffix,
    block_size,
    output_dtype,
    workspace=None,
):
    """Run representative + selected-page attention without condition selection."""
    if workspace is None:
        workspace = {}
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        raise ValueError("fixed-mask attention requires q_len=1")
    n_blocks = int(prompt_prefix["block_valid_counts"].numel())
    if selected.shape != (n_kv_heads, n_blocks):
        raise ValueError(
            f"selected must have shape {(n_kv_heads, n_blocks)}, got {selected.shape}"
        )
    block_n = 32
    n_chunks = triton.cdiv(n_blocks, block_n)
    rows = n_kv_heads * group_size
    q = q_grouped.reshape(rows, head_dim).contiguous()
    partial_acc = _workspace_tensor(
        workspace,
        "partial_acc",
        (rows, n_chunks, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    partial_m = _workspace_tensor(
        workspace,
        "partial_m",
        (rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )
    partial_l = _workspace_tensor(
        workspace,
        "partial_l",
        (rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )
    output = _workspace_tensor(
        workspace,
        "output",
        (rows, head_dim),
        device=q.device,
        dtype=output_dtype,
    )
    k_block = prompt_prefix["k_block_attn"]
    v_block = prompt_prefix["v_block_attn"]
    _fixed_mask_hybrid_attention_kernel[(n_kv_heads, n_chunks)](
        q,
        k_block,
        v_block,
        prompt_prefix["k_bar"].contiguous(),
        prompt_prefix["v_bar"].contiguous(),
        prompt_prefix["block_valid_counts"].contiguous(),
        selected.contiguous(),
        k_suffix,
        v_suffix,
        partial_acc,
        partial_m,
        partial_l,
        n_blocks,
        int(k_suffix.shape[1]),
        k_block.stride(0),
        v_block.stride(0),
        k_suffix.stride(0),
        k_suffix.stride(1),
        v_suffix.stride(0),
        v_suffix.stride(1),
        group_size,
        head_dim,
        n_chunks,
        head_dim**-0.5,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        PAGE_SIZE=int(block_size),
        num_warps=4,
    )
    _condition_block_stage2_reduce_kernel[(rows,)](
        partial_acc,
        partial_m,
        partial_l,
        output,
        n_chunks,
        head_dim,
        BLOCK_C=triton.next_power_of_2(n_chunks),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output.reshape(n_kv_heads, group_size, 1, head_dim)
