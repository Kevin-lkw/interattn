"""Synthetic multi-query selection kernel used to measure IO amortization."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def condition_block_selection_stats_multi_query_kernel(
    q_ptr,
    k_bar_ptr,
    k_bounds_desc,
    counts_ptr,
    s_cache_ptr,
    delta_cache_ptr,
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    n_blocks,
    n_chunks,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reuse one summary tile across K * G query rows for one KV head."""
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    r = tl.arange(0, BLOCK_R)
    b = chunk * BLOCK_B + tl.arange(0, BLOCK_B)
    d = tl.arange(0, BLOCK_D)
    r_mask = r < QUERY_ROWS
    b_mask = b < n_blocks
    d_mask = d < head_dim
    row = kv_head * QUERY_ROWS + r

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d[None, :],
        mask=r_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
    k_bar = tl.load(
        k_bar_ptr + stat_off,
        mask=active[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    bounds = tl.load_tensor_descriptor(
        k_bounds_desc,
        [kv_head, chunk * BLOCK_B, 0],
    ).to(tl.float32)
    bounds = tl.reshape(bounds, (BLOCK_B, BLOCK_D, 2))
    k_hi, k_lo = tl.split(bounds)

    s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
    upper_k = tl.where(q[:, None, :] >= 0.0, k_hi[None, :, :], k_lo[None, :, :])
    lower_k = tl.where(q[:, None, :] >= 0.0, k_lo[None, :, :], k_hi[None, :, :])
    upper = tl.sum(q[:, None, :] * upper_k, axis=2) * scale
    lower = tl.sum(q[:, None, :] * lower_k, axis=2) * scale
    delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))
    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    active_2d = r_mask[:, None] & active[None, :]
    z = tl.where(active_2d, z, -float("inf"))
    cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
    zc = z + tl.log(cosh_delta)

    tl.store(
        s_cache_ptr + row[:, None] * n_blocks + b[None, :],
        s,
        mask=active_2d,
    )
    tl.store(
        delta_cache_ptr + row[:, None] * n_blocks + b[None, :],
        delta,
        mask=active_2d,
    )
    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=r_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=r_mask)
    tl.store(c_m_ptr + partial_off, c_m, mask=r_mask)
    tl.store(c_l_ptr + partial_off, c_l, mask=r_mask)
