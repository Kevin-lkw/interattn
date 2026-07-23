"""Triton selection-stats kernel for the diag_ell condition (strict bound).

Drop-in replacement for the production box selection-stats kernel
(`core._condition_block_selection_stats_kernel`): identical outputs
(`s`/`delta` caches plus the four per-chunk softmax partials), but per block it
reads `k_bar` + `w` (1 vector) + `rho` (1 scalar) instead of
`k_bar` + `k_max` + `k_min` (3 vectors) — a ~1.5x summary-read reduction. The
downstream finalize/stage2 kernels consume `s`/`delta` from cache and need no
changes.

delta = rho * ||q * w|| * scale, with scale = 1/sqrt(head_dim); everything
after the delta computation is copied verbatim from the box kernel.
"""

import torch
import triton
import triton.language as tl

from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import diag_ell_stats


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _condition_block_selection_stats_diag_ell_kernel(
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
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
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
    w = tl.load(w_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
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
    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=g_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=g_mask)
    tl.store(c_m_ptr + partial_off, c_m, mask=g_mask)
    tl.store(c_l_ptr + partial_off, c_l, mask=g_mask)


def run_selection_stats_diag_ell(
    q_grouped,
    prefix,
    workspace=None,
    reduce_globals=True,
    term1_mass_exp=False,
):
    """Mirror of `core._run_condition_block_selection_stats` for diag_ell.

    Requires `w`/`rho` in the prefix (computed once per layer by
    `gen_selection.diag_ell_stats`).
    """
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        raise ValueError("Triton selection stats expects q_len=1.")
    w, rho = diag_ell_stats(prefix)
    n_blocks = int(prefix["block_valid_counts"].numel())
    rows = int(n_kv_heads * group_size)
    q = q_grouped.reshape(rows, head_dim).contiguous()
    selection_chunk = core._SELECT_CHUNK
    n_chunks = triton.cdiv(n_blocks, selection_chunk)
    s_cache = core._workspace_empty(
        workspace, "selection_s", (rows, n_blocks), device=q.device, dtype=torch.float32
    )
    delta_cache = core._workspace_empty(
        workspace, "selection_delta", (rows, n_blocks), device=q.device, dtype=torch.float32
    )
    partial = core._workspace_empty(
        workspace, "selection_partial", (4, rows, n_chunks), device=q.device, dtype=torch.float32
    )
    _condition_block_selection_stats_diag_ell_kernel[(n_kv_heads, n_chunks)](
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
        group_size,
        head_dim,
        head_dim**-0.5,
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_B=selection_chunk,
        BLOCK_D=triton.next_power_of_2(head_dim),
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=core._select_warps(n_blocks),
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
            n_chunks,
            BLOCK_C=triton.next_power_of_2(n_chunks),
            num_warps=4,
        )
    return q, s_cache, delta_cache, partial, global_stats, n_blocks, n_chunks
