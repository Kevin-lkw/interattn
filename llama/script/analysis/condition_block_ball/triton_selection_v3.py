"""Tensor-core diag_ell selection-stats kernel (v3, strict bound kept).

The v1/v2 kernels compute `s` and `q^2.w^2` with broadcast-multiply-sum, whose
register footprint scales with BLOCK_B x BLOCK_D — that caps the tile at 16
blocks and forbids deep pipelining (measured: BLOCK_B=32 spills, persistent v2
only ties v1). v3 reformulates both contractions as `tl.dot` MMA so the
footprint is fixed and small, enabling BLOCK_B=64/128 tiles and a
software-pipelined persistent loop — the flash-style shape.

Strictness: the summary is `w2 = bf16(w*w)` (any positive weight keeps the
w-weighted Cauchy-Schwarz bound valid) and `rho` is computed against the
*stored* w2, so the only kernel-side underestimate is the BF16 rounding of
q^2 (<= 2^-9 relative) plus FP32 accumulation slop; the delta scale carries a
(1 + 2^-8) inflation that strictly covers both. Note vs v1 the delta is not
one-sidedly larger: v1's round-up `w` and v3's round-to-nearest `w2` are
*different* valid weights, so per-block deltas differ ~2^-8 in either
direction (borderline selections can flip both ways; soundness holds for
each independently and is asserted directly in `bench_selection_v3.py`).

MMA needs M >= 16, so the 4 group rows are padded to 16 with fully masked
lanes. Requires BF16 `k_bar` (CONDITION_BLOCK_K_BAR_DTYPE=bfloat16 +
CONDITION_BLOCK_MIXED_SUMMARIES=1).

Env knobs: CONDITION_BLOCK_SELECT_V3_BLOCK (default 64), _P (default 48),
_STAGES (default 3), _WARPS (default 4).
"""

import os

import torch
import triton
import triton.language as tl

from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import _masked_weighted_radius

_V3_BLOCK = int(os.environ.get("CONDITION_BLOCK_SELECT_V3_BLOCK", "32"))
_V3_P = int(os.environ.get("CONDITION_BLOCK_SELECT_V3_P", "48"))
_V3_STAGES = int(os.environ.get("CONDITION_BLOCK_SELECT_V3_STAGES", "3"))
_V3_WARPS = int(os.environ.get("CONDITION_BLOCK_SELECT_V3_WARPS", "4"))

_DELTA_INFLATION = 1.0 + 2.0**-8


def diag_ell_v3_stats(prefix):
    """`w2` (BF16, elementwise squared deviation bound) + `rho` from stored w2."""
    stats = prefix.get("diag_ell_v3_stats")
    if stats is None:
        k_bar = prefix["k_bar"].float()
        w = torch.maximum(
            prefix["k_max"].float() - k_bar,
            k_bar - prefix["k_min"].float(),
        ).clamp_min(1e-6)
        w2 = (w * w).to(torch.bfloat16)
        rho = _masked_weighted_radius(prefix, inv_w2=w2.float().reciprocal())
        stats = (w2, rho)
        prefix["diag_ell_v3_stats"] = stats
    return stats


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks", "span", "num_p"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks", "span", "num_p"],
)
def _condition_block_selection_stats_v3_kernel(
    q_ptr,
    k_bar_ptr,
    w2_ptr,
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
    delta_scale: tl.constexpr,
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
    q_bf = tl.load(
        q_ptr + row[:, None] * head_dim + d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    q_f32 = q_bf.to(tl.float32)
    q2_bf = (q_f32 * q_f32).to(tl.bfloat16)

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
        # (D, B) tiles: d fastest -> each block's 128 values are one
        # contiguous 256 B run; dot(q (G,D), tile (D,B)) -> (G, B).
        tile_off = (kv_head * n_blocks + b[None, :]) * head_dim + d[:, None]
        tile_mask = active[None, :] & d_mask[:, None]
        k_bar = tl.load(k_bar_ptr + tile_off, mask=tile_mask, other=0.0)
        w2 = tl.load(w2_ptr + tile_off, mask=tile_mask, other=0.0)
        rho = tl.load(rho_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)
        s = tl.dot(q_bf, k_bar) * scale
        qw2 = tl.dot(q2_bf, w2)
        delta = rho[None, :] * tl.sqrt(tl.maximum(qw2, 0.0)) * delta_scale
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


def run_selection_stats_diag_ell_v3(
    q_grouped,
    prefix,
    workspace=None,
    reduce_globals=True,
    term1_mass_exp=False,
):
    """Drop-in mirror of `run_selection_stats_diag_ell` (tensor-core version)."""
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    if n_query != 1:
        raise ValueError("Triton selection stats expects q_len=1.")
    if prefix["k_bar"].dtype != torch.bfloat16:
        raise ValueError("v3 requires BF16 k_bar (CONDITION_BLOCK_K_BAR_DTYPE=bfloat16).")
    w2, rho = diag_ell_v3_stats(prefix)
    n_blocks = int(prefix["block_valid_counts"].numel())
    rows = int(n_kv_heads * group_size)
    q = q_grouped.reshape(rows, head_dim).contiguous()
    n_chunks = triton.cdiv(n_blocks, _V3_BLOCK)
    num_p = min(_V3_P, n_chunks)
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
    _condition_block_selection_stats_v3_kernel[(n_kv_heads, num_p)](
        q,
        prefix["k_bar"].contiguous(),
        w2.contiguous(),
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
        head_dim**-0.5 * _DELTA_INFLATION,
        BLOCK_G=16,
        BLOCK_B=_V3_BLOCK,
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_STAGES=_V3_STAGES,
        TERM1_MASS_EXP=bool(term1_mass_exp),
        num_warps=_V3_WARPS,
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
