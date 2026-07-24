"""Persistent finalize-attention kernel (v2) + fused stage2 runner.

The production finalize kernel already does its heavy math with `tl.dot`,
but its grid is one short-lived program per 32-block chunk (~13 KB of reads
each: v_bar tile 8 KB + s/delta 1 KB + an inline re-reduce of the selection
partials repeated by *every* program), and the entire generated suffix is
processed serially by the last chunk's single program. Measured ~30% of its
byte bound. This restructure applies the v3 surgery:

- grid (kv_heads, P) persistent programs; each streams a contiguous span of
  chunks in a pipelined `tl.range` loop, carrying the online softmax
  (m, l, acc) across the span and writing ONE partial per program (the
  stage2 reduce shrinks from n_chunks to P inputs);
- the selection-partial re-reduce happens once per program (P times per
  head) instead of once per chunk;
- suffix tiles are distributed round-robin over the P programs and merged
  into the same running softmax — no serial tail;
- selection math is unchanged: per-block decisions are bitwise identical to
  the production kernel (same s/delta/globals, same reduce tree).

Runner `decode_output_fused_v2` is a drop-in replacement for
`core._condition_block_decode_output_fused_triton`, with the stats stage
served by the v3 tensor-core kernel.

Env knobs: CONDITION_BLOCK_FIN_V2_P (default 48), _BLOCK (chunk width,
default 32), _STAGES (default 2), _WARPS (default 4).
"""

import os

import torch
import triton
import triton.language as tl

from ..condition_block_gen.methods.condition_block_triton_impl import core
from .triton_selection_v3 import run_selection_stats_diag_ell_v3

_FIN_P = int(os.environ.get("CONDITION_BLOCK_FIN_V2_P", "128"))
_FIN_BLOCK = int(os.environ.get("CONDITION_BLOCK_FIN_V2_BLOCK", "32"))
_FIN_STAGES = int(os.environ.get("CONDITION_BLOCK_FIN_V2_STAGES", "2"))
_FIN_WARPS = int(os.environ.get("CONDITION_BLOCK_FIN_V2_WARPS", "4"))


@triton.jit(
    do_not_specialize=[
        "n_blocks",
        "n_sel_chunks",
        "n_chunks",
        "span",
        "num_pf",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
    do_not_specialize_on_alignment=[
        "n_blocks",
        "n_sel_chunks",
        "n_chunks",
        "span",
        "num_pf",
        "k_block_head_stride",
        "v_block_head_stride",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
)
def _condition_block_finalize_attention_v2_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    v_bar_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    v_norm_ptr,
    v_norm_all_ptr,
    counts_ptr,
    z_m_part_ptr,
    z_l_part_ptr,
    c_m_part_ptr,
    c_l_part_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    selected_out_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    n_sel_chunks,
    n_chunks,
    span,
    num_pf,
    suffix_len_ptr,
    k_block_head_stride,
    v_block_head_stride,
    k_suffix_head_stride,
    k_suffix_token_stride,
    v_suffix_head_stride,
    v_suffix_token_stride,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SC: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    STORE_SELECTED: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
):
    kv_head = tl.program_id(0)
    p = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    p_off = tl.arange(0, PAGE_SIZE)
    m_mask = m_off < group_size
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    # Selection-partial re-reduce: once per program (the production kernel
    # repeats this in every chunk program).
    sc_off = tl.arange(0, BLOCK_SC)
    sc_mask = sc_off < n_sel_chunks
    part_off = row[:, None] * n_sel_chunks + sc_off[None, :]
    part_mask = m_mask[:, None] & sc_mask[None, :]
    z_m_part = tl.load(z_m_part_ptr + part_off, mask=part_mask, other=-float("inf"))
    z_l_part = tl.load(z_l_part_ptr + part_off, mask=part_mask, other=0.0)
    c_m_part = tl.load(c_m_part_ptr + part_off, mask=part_mask, other=-float("inf"))
    c_l_part = tl.load(c_l_part_ptr + part_off, mask=part_mask, other=0.0)
    z_m = tl.max(z_m_part, axis=1)
    c_m = tl.max(c_m_part, axis=1)
    z_l = tl.sum(z_l_part * tl.exp(z_m_part - z_m[:, None]), axis=1)
    c_l = tl.sum(c_l_part * tl.exp(c_m_part - c_m[:, None]), axis=1)
    z_m = tl.where(m_mask, z_m, 0.0)
    z_l = tl.where(m_mask, z_l, 1.0)
    c_m = tl.where(m_mask, c_m, 0.0)
    c_l = tl.where(m_mask, c_l, 1.0)
    b_all = tl.load(v_norm_all_ptr + kv_head).to(tl.float32)

    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for i in tl.range(0, span, num_stages=NUM_STAGES):
        chunk = p * span + i
        block_idx = chunk * BLOCK_N + n_off
        block_mask = block_idx < n_blocks
        mask_2d = m_mask[:, None] & block_mask[None, :]
        count = tl.load(counts_ptr + block_idx, mask=block_mask, other=0)
        active_block = block_mask & (count > 0)
        active_2d = m_mask[:, None] & active_block[None, :]
        s = tl.load(
            s_cache_ptr + row[:, None] * n_blocks + block_idx[None, :],
            mask=active_2d,
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            delta_cache_ptr + row[:, None] * n_blocks + block_idx[None, :],
            mask=active_2d,
            other=0.0,
        ).to(tl.float32)
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]

        b_c = tl.load(
            v_norm_ptr + kv_head * n_blocks + block_idx,
            mask=active_block,
            other=0.0,
        ).to(tl.float32)
        if TERM1_MASS_EXP:
            exp_neg_delta = tl.exp(-delta)
            tanh_half = (1.0 - exp_neg_delta) / (1.0 + exp_neg_delta)
            term1 = 2.0 * b_all * tl.exp(z + delta - c_m[:, None]) / c_l[:, None]
        else:
            cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
            tanh_half = 2.0 / (1.0 + tl.exp(-delta)) - 1.0
            term1 = 2.0 * b_all * tl.exp(z - c_m[:, None]) * (cosh_delta - 1.0) / c_l[:, None]
        term2 = 2.0 * b_c[None, :] * tl.exp(z - z_m[:, None]) * tanh_half / z_l[:, None]
        condition = tl.where(active_2d, term1 + term2, 0.0)
        selected = ((tl.sum(condition, axis=0) / group_size) > eps) & active_block
        if STORE_SELECTED:
            tl.store(
                selected_out_ptr + row[:, None] * n_blocks + block_idx[None, :],
                selected[None, :],
                mask=mask_2d,
            )

        active_rep = active_block & (~selected)
        rep_scores = tl.where(m_mask[:, None] & active_rep[None, :], z, -float("inf"))
        has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
        rep_m = tl.max(rep_scores, axis=1)
        new_m = tl.where(has_rep & m_mask, tl.maximum(softmax_m, rep_m), softmax_m)
        alpha = tl.where(has_rep & m_mask, tl.exp(softmax_m - new_m), 1.0)
        beta = tl.where(
            m_mask[:, None] & active_rep[None, :],
            tl.exp(rep_scores - new_m[:, None]),
            0.0,
        )
        rep_off = (kv_head * n_blocks + block_idx[:, None]) * head_dim + d_off[None, :]
        v_rep = tl.load(
            v_bar_ptr + rep_off,
            mask=active_rep[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v_rep, out_dtype=tl.float32)
        softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
        softmax_m = new_m

        sel_cum = tl.cumsum(selected.to(tl.int32), axis=0)
        n_sel = tl.sum(selected.to(tl.int32), axis=0)
        sel_j = 0
        while sel_j < n_sel:
            page_local = tl.sum(tl.where((sel_cum == sel_j + 1) & selected, n_off, 0), axis=0)
            page = chunk * BLOCK_N + page_local
            page_count = tl.load(counts_ptr + page)
            token_active = p_off < page_count
            token_off = (page * PAGE_SIZE + p_off[:, None]) * head_dim + d_off[None, :]
            page_k = tl.load(
                k_block_ptr + kv_head * k_block_head_stride + token_off,
                mask=token_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            page_scores = tl.dot(q, tl.trans(page_k), out_dtype=tl.float32) * scale
            page_scores = tl.where(
                m_mask[:, None] & token_active[None, :], page_scores, -float("inf")
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

    # Suffix tiles are distributed round-robin over the P programs instead of
    # being serialized in the last chunk's program.
    suffix_len = tl.load(suffix_len_ptr)
    suffix_start = p * BLOCK_N
    while suffix_start < suffix_len:
        suffix_idx = suffix_start + n_off
        suffix_active = suffix_idx < suffix_len
        k_suffix_off = (
            kv_head * k_suffix_head_stride
            + suffix_idx[:, None] * k_suffix_token_stride
            + d_off[None, :]
        )
        k = tl.load(
            k_suffix_ptr + k_suffix_off,
            mask=suffix_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(m_mask[:, None] & suffix_active[None, :], scores, -float("inf"))
        tile_m = tl.max(scores, axis=1)
        new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
        alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
        beta = tl.where(
            m_mask[:, None] & suffix_active[None, :],
            tl.exp(scores - new_m[:, None]),
            0.0,
        )
        v_suffix_off = (
            kv_head * v_suffix_head_stride
            + suffix_idx[:, None] * v_suffix_token_stride
            + d_off[None, :]
        )
        v = tl.load(
            v_suffix_ptr + v_suffix_off,
            mask=suffix_active[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.bfloat16), v, out_dtype=tl.float32)
        softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
        softmax_m = new_m
        suffix_start += num_pf * BLOCK_N

    partial_off = row * num_pf + p
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


def decode_output_fused_v2(
    *,
    q_grouped,
    prompt_prefix,
    k_suffix,
    v_suffix,
    suffix_len_dev,
    eps,
    page_size,
    store_selected,
    output_dtype,
    workspace=None,
    term1_mass_exp=False,
):
    """Drop-in replacement for `core._condition_block_decode_output_fused_triton`."""
    n_kv_heads, group_size, _n_query, head_dim = q_grouped.shape
    q, s_cache, delta_cache, sel_partial, _global_stats, n_blocks, n_sel_chunks = (
        run_selection_stats_diag_ell_v3(
            q_grouped,
            prompt_prefix,
            workspace,
            reduce_globals=False,
            term1_mass_exp=term1_mass_exp,
        )
    )
    rows = int(n_kv_heads * group_size)
    n_chunks = triton.cdiv(n_blocks, _FIN_BLOCK)
    num_pf = min(_FIN_P, n_chunks)
    span = triton.cdiv(n_chunks, num_pf)
    partial_acc = core._workspace_empty(
        workspace, "attention_partial_acc", (rows, num_pf, head_dim),
        device=q.device, dtype=torch.float32,
    )
    partial_m = core._workspace_empty(
        workspace, "attention_partial_m", (rows, num_pf), device=q.device, dtype=torch.float32
    )
    partial_l = core._workspace_empty(
        workspace, "attention_partial_l", (rows, num_pf), device=q.device, dtype=torch.float32
    )
    selected_rows = core._workspace_empty(
        workspace,
        "selected_rows" if store_selected else "selected_dummy",
        (rows, n_blocks) if store_selected else (1,),
        device=q.device,
        dtype=torch.bool,
    )
    k_block = prompt_prefix["k_block_attn"]
    v_block = prompt_prefix["v_block_attn"]
    _condition_block_finalize_attention_v2_kernel[(n_kv_heads, num_pf)](
        q,
        k_block,
        v_block,
        prompt_prefix["v_bar"].contiguous(),
        s_cache,
        delta_cache,
        prompt_prefix["v_norm_max"].contiguous(),
        prompt_prefix["v_norm_all"].contiguous(),
        prompt_prefix["block_valid_counts"].contiguous(),
        sel_partial[0],
        sel_partial[1],
        sel_partial[2],
        sel_partial[3],
        k_suffix,
        v_suffix,
        selected_rows,
        partial_acc,
        partial_m,
        partial_l,
        n_blocks,
        n_sel_chunks,
        n_chunks,
        span,
        num_pf,
        suffix_len_dev,
        k_block.stride(0),
        v_block.stride(0),
        k_suffix.stride(0),
        k_suffix.stride(1),
        v_suffix.stride(0),
        v_suffix.stride(1),
        group_size,
        head_dim,
        float(eps),
        head_dim**-0.5,
        BLOCK_M=16,
        BLOCK_N=_FIN_BLOCK,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_SC=triton.next_power_of_2(n_sel_chunks),
        PAGE_SIZE=int(page_size),
        NUM_STAGES=_FIN_STAGES,
        STORE_SELECTED=bool(store_selected),
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=_FIN_WARPS,
    )
    out = core._workspace_empty(
        workspace, "attention_output", (rows, head_dim), device=q.device, dtype=output_dtype
    )
    core._condition_block_stage2_reduce_kernel[(rows,)](
        partial_acc,
        partial_m,
        partial_l,
        out,
        num_pf,
        head_dim,
        BLOCK_C=triton.next_power_of_2(num_pf),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    selected = None
    if store_selected:
        selected = selected_rows.reshape(n_kv_heads, group_size, 1, n_blocks)
    return out.reshape(n_kv_heads, group_size, 1, head_dim), selected
