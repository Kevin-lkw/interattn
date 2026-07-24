"""Persistent diag_ell selection-stats kernel (v2).

Same math and same `s`/`delta` outputs (bitwise) as `triton_selection.py`, but
restructured for bandwidth efficiency: instead of one short-lived program per
16-block chunk (8 KB of reads each, nothing to hide memory latency behind),
the grid is (kv_heads, P) persistent programs. Each program streams a
*contiguous* span of chunks in a software-pipelined `tl.range` loop and merges
the four softmax partials online, so it writes one partial per program: the
partial count drops from n_chunks (256 at 128K) to P (default 48), which also
shrinks the inline re-reduce every finalize program performs over
`sel_partial` (BLOCK_SC 256 -> 64).

Env knobs: CONDITION_BLOCK_SELECT_V2_P (default 48),
CONDITION_BLOCK_SELECT_V2_BLOCK (inner tile, default 32),
CONDITION_BLOCK_SELECT_V2_STAGES (pipeline stages, default 3),
CONDITION_BLOCK_SELECT_V2_WARPS (default 4).
"""

import os

import torch
import triton
import triton.language as tl

from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import diag_ell_stats

_V2_P = int(os.environ.get("CONDITION_BLOCK_SELECT_V2_P", "48"))
_V2_BLOCK = int(os.environ.get("CONDITION_BLOCK_SELECT_V2_BLOCK", "32"))
_V2_STAGES = int(os.environ.get("CONDITION_BLOCK_SELECT_V2_STAGES", "3"))
_V2_WARPS = int(os.environ.get("CONDITION_BLOCK_SELECT_V2_WARPS", "4"))


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks", "span", "num_p"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks", "span", "num_p"],
)
def _condition_block_selection_stats_v2_kernel(
    q_ptr,
    k_bar_ptr,
    w_ptr,
    rho_ptr,
    counts_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    n_blocks,
    n_chunks,
    span,
    num_p,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
):
    kv_head = tl.program_id(0)
    p = tl.program_id(1)
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

    run_zm = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    run_zl = tl.zeros((BLOCK_G,), tl.float32)
    run_cm = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    run_cl = tl.zeros((BLOCK_G,), tl.float32)

    for i in tl.range(0, span, num_stages=NUM_STAGES):
        chunk = p * span + i
        b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b < n_blocks
        count = tl.load(counts_ptr + b, mask=b_mask, other=0)
        active = b_mask & (count > 0)
        stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
        load_mask = active[:, None] & d_mask[None, :]
        k_bar = tl.load(k_bar_ptr + stat_off, mask=load_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + stat_off, mask=load_mask, other=0.0).to(tl.float32)
        rho = tl.load(rho_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
        qw2 = tl.sum((q * q)[:, None, :] * (w * w)[None, :, :], axis=2)
        delta = rho[None, :] * tl.sqrt(qw2) * scale
        z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
        active_2d = g_mask[:, None] & active[None, :]
        z = tl.where(active_2d, z, -float("inf"))
        if TERM1_MASS_EXP:
            zc = z + delta
        else:
            cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
            zc = z + tl.log(cosh_delta)
        tl.store(s_cache_ptr + row[:, None] * n_blocks + b[None, :], s, mask=active_2d)
        tl.store(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], delta, mask=active_2d)

        # Online merge of the chunk into the running (m, l) pairs. The safe
        # base keeps exp() away from (-inf) - (-inf) when a whole span tail is
        # out of range: contributions from -inf logits are exactly 0.
        new_zm = tl.maximum(run_zm, tl.max(z, axis=1))
        zm_safe = tl.where(new_zm > -float("inf"), new_zm, 0.0)
        run_zl = run_zl * tl.exp(run_zm - zm_safe) + tl.sum(tl.exp(z - zm_safe[:, None]), axis=1)
        run_zm = new_zm
        new_cm = tl.maximum(run_cm, tl.max(zc, axis=1))
        cm_safe = tl.where(new_cm > -float("inf"), new_cm, 0.0)
        run_cl = run_cl * tl.exp(run_cm - cm_safe) + tl.sum(tl.exp(zc - cm_safe[:, None]), axis=1)
        run_cm = new_cm

    partial_off = row * num_p + p
    tl.store(z_m_ptr + partial_off, run_zm, mask=g_mask)
    tl.store(z_l_ptr + partial_off, run_zl, mask=g_mask)
    tl.store(c_m_ptr + partial_off, run_cm, mask=g_mask)
    tl.store(c_l_ptr + partial_off, run_cl, mask=g_mask)


def run_selection_stats_diag_ell_v2(
    q_grouped,
    prefix,
    workspace=None,
    reduce_globals=True,
    term1_mass_exp=False,
):
    """Drop-in mirror of `run_selection_stats_diag_ell` with the persistent grid.

    Returns P as the chunk count, so the production finalize kernel re-reduces
    P partials instead of n_chunks.
    """
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        raise ValueError("Triton selection stats expects q_len=1.")
    w, rho = diag_ell_stats(prefix)
    n_blocks = int(prefix["block_valid_counts"].numel())
    rows = int(n_kv_heads * group_size)
    q = q_grouped.reshape(rows, head_dim).contiguous()
    n_chunks = triton.cdiv(n_blocks, _V2_BLOCK)
    num_p = min(_V2_P, n_chunks)
    span = triton.cdiv(n_chunks, num_p)
    s_cache = core._workspace_empty(
        workspace, "selection_s", (rows, n_blocks), device=q.device, dtype=torch.float32
    )
    delta_cache = core._workspace_empty(
        workspace, "selection_delta", (rows, n_blocks), device=q.device, dtype=torch.float32
    )
    partial = core._workspace_empty(
        workspace, "selection_partial", (4, rows, num_p), device=q.device, dtype=torch.float32
    )
    _condition_block_selection_stats_v2_kernel[(n_kv_heads, num_p)](
        q,
        prefix["k_bar"].contiguous(),
        w.contiguous(),
        rho.contiguous(),
        prefix["block_valid_counts"].contiguous(),
        s_cache,
        delta_cache,
        partial[0],
        partial[1],
        partial[2],
        partial[3],
        n_blocks,
        n_chunks,
        span,
        num_p,
        group_size,
        head_dim,
        head_dim**-0.5,
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_B=_V2_BLOCK,
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_STAGES=_V2_STAGES,
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=_V2_WARPS,
    )
    global_stats = None
    if reduce_globals:
        global_stats = core._workspace_empty(
            workspace, "selection_global", (4, rows), device=q.device, dtype=torch.float32
        )
        core._condition_block_selection_reduce_kernel[(rows,)](
            partial[0],
            partial[1],
            partial[2],
            partial[3],
            global_stats[0],
            global_stats[1],
            global_stats[2],
            global_stats[3],
            num_p,
            BLOCK_C=triton.next_power_of_2(num_p),
            num_warps=4,
        )
    return q, s_cache, delta_cache, partial, global_stats, n_blocks, num_p
