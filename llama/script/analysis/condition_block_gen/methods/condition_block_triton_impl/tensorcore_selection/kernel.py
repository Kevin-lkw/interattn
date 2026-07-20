"""Tensor Core prototype for condition-block range-bound statistics.

The formulation is identical to the production coordinate range bound.  The
kernel rewrites the sign-dependent reductions as

    upper = q_pos @ k_max + q_neg @ k_min
    lower = q_pos @ k_min + q_neg @ k_max

and places ``q``, ``q_pos`` and ``q_neg`` in one padded 16-row MMA tile.  The
three summary matrices are assembled as one logical 64-column tile in
registers, so each summary value is still loaded only once from global memory.

This module is deliberately independent from ``core.py``.  It is an
experimental stage kernel until both numerical and latency gates pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_PRECISIONS = {"ieee", "tf32x3", "tf32"}


def _workspace_empty(workspace, key, shape, *, device, dtype):
    if workspace is None:
        return torch.empty(shape, device=device, dtype=dtype)
    tensor = workspace.get(key)
    if (
        tensor is None
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        workspace[key] = tensor
    return tensor


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _selection_stats_tensorcore_kernel(
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
    n_blocks,
    n_chunks,
    scale: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    TERM1_MASS_EXP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute exact coordinate bounds through a padded 16x128x64 MMA.

    The logical row groups are ``[q, q_pos, q_neg, zero]`` and the logical
    column groups are ``[k_bar, k_max, k_min, zero]``.  The useful submatrices
    of the product recover s, upper and lower without changing the bound.
    """

    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)

    g = tl.arange(0, GROUP_SIZE)
    b_lane = tl.arange(0, BLOCK_B)
    d = tl.arange(0, BLOCK_D)
    row_group = tl.arange(0, 4)
    col_group = tl.arange(0, 4)

    b = chunk * BLOCK_B + b_lane
    b_mask = b < n_blocks
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)

    row = kv_head * GROUP_SIZE + g
    q = tl.load(
        q_ptr + row[:, None] * BLOCK_D + d[None, :]
    ).to(tl.float32)
    q_pos = tl.maximum(q, 0.0)
    q_neg = tl.minimum(q, 0.0)

    # [4, G, D] -> [16, D].  GROUP_SIZE=4 is the Llama-3.1-8B GQA layout.
    q_groups = tl.where(
        row_group[:, None, None] == 0,
        q[None, :, :],
        tl.where(
            row_group[:, None, None] == 1,
            q_pos[None, :, :],
            tl.where(row_group[:, None, None] == 2, q_neg[None, :, :], 0.0),
        ),
    )
    q_mma = tl.reshape(q_groups, (4 * GROUP_SIZE, BLOCK_D))

    stat_off = ((kv_head * n_blocks + b[:, None]) * BLOCK_D) + d[None, :]
    stat_mask = active[:, None]
    k_bar = tl.load(k_bar_ptr + stat_off, mask=stat_mask, other=0.0).to(tl.float32)
    k_max = tl.load(k_max_ptr + stat_off, mask=stat_mask, other=0.0).to(tl.float32)
    k_min = tl.load(k_min_ptr + stat_off, mask=stat_mask, other=0.0).to(tl.float32)
    k_bar_t = tl.trans(k_bar)
    k_max_t = tl.trans(k_max)
    k_min_t = tl.trans(k_min)

    # [D, 4, B] -> [D, 64].  The fourth group is padding for the MMA tile.
    k_groups = tl.where(
        col_group[None, :, None] == 0,
        k_bar_t[:, None, :],
        tl.where(
            col_group[None, :, None] == 1,
            k_max_t[:, None, :],
            tl.where(col_group[None, :, None] == 2, k_min_t[:, None, :], 0.0),
        ),
    )
    k_mma = tl.reshape(k_groups, (BLOCK_D, 4 * BLOCK_B))

    products = tl.dot(
        q_mma,
        k_mma,
        input_precision=DOT_PRECISION,
        out_dtype=tl.float32,
    )
    products = tl.reshape(
        products,
        (4, GROUP_SIZE, 4, BLOCK_B),
    )
    rg = row_group[:, None, None, None]
    cg = col_group[None, None, :, None]

    s_terms = tl.where((rg == 0) & (cg == 0), products, 0.0)
    upper_terms = tl.where(
        ((rg == 1) & (cg == 1)) | ((rg == 2) & (cg == 2)),
        products,
        0.0,
    )
    lower_terms = tl.where(
        ((rg == 1) & (cg == 2)) | ((rg == 2) & (cg == 1)),
        products,
        0.0,
    )
    s = tl.sum(tl.sum(s_terms, axis=2), axis=0) * scale
    upper = tl.sum(tl.sum(upper_terms, axis=2), axis=0) * scale
    lower = tl.sum(tl.sum(lower_terms, axis=2), axis=0) * scale
    delta = tl.maximum(tl.abs(upper - s), tl.abs(lower - s))

    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    active_2d = active[None, :]
    z = tl.where(active_2d, z, -float("inf"))
    if TERM1_MASS_EXP:
        zc = z + delta
    else:
        cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
        zc = z + tl.log(cosh_delta)

    row_out = kv_head * GROUP_SIZE + g
    tl.store(
        s_cache_ptr + row_out[:, None] * n_blocks + b[None, :],
        s,
        mask=active_2d,
    )
    tl.store(
        delta_cache_ptr + row_out[:, None] * n_blocks + b[None, :],
        delta,
        mask=active_2d,
    )

    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row_out * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m)
    tl.store(z_l_ptr + partial_off, z_l)
    tl.store(c_m_ptr + partial_off, c_m)
    tl.store(c_l_ptr + partial_off, c_l)


@triton.jit(
    do_not_specialize=["n_chunks"],
    do_not_specialize_on_alignment=["n_chunks"],
)
def _selection_reduce_kernel(
    z_m_ptr,
    z_l_ptr,
    c_m_ptr,
    c_l_ptr,
    global_z_m_ptr,
    global_z_l_ptr,
    global_c_m_ptr,
    global_c_l_ptr,
    n_chunks,
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


def run_tensorcore_selection_stats(
    q_grouped,
    prefix,
    workspace=None,
    *,
    precision="ieee",
    reduce_globals=True,
    term1_mass_exp=False,
    num_warps=4,
):
    """Run the experimental Tensor Core stats kernel.

    The prototype intentionally targets the production Llama-3.1-8B geometry:
    four Q heads per KV head, head dimension 128, and a 16-block selection
    chunk.  Unsupported layouts fail loudly instead of silently falling back.
    """

    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(_SUPPORTED_PRECISIONS)}, got {precision!r}"
        )
    n_kv_heads, group_size, n_query, head_dim = map(int, q_grouped.shape)
    if n_query != 1 or group_size != 4 or head_dim != 128:
        raise ValueError(
            "Tensor Core selection prototype requires q shape "
            "[n_kv_heads, 4, 1, 128]."
        )
    n_blocks = int(prefix["block_valid_counts"].numel())
    block_b = 16
    n_chunks = triton.cdiv(n_blocks, block_b)
    rows = n_kv_heads * group_size
    q = q_grouped.reshape(rows, head_dim).contiguous()

    s_cache = _workspace_empty(
        workspace,
        "tc_selection_s",
        (rows, n_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    delta_cache = _workspace_empty(
        workspace,
        "tc_selection_delta",
        (rows, n_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    partial = _workspace_empty(
        workspace,
        "tc_selection_partial",
        (4, rows, n_chunks),
        device=q.device,
        dtype=torch.float32,
    )

    _selection_stats_tensorcore_kernel[(n_kv_heads, n_chunks)](
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
        head_dim**-0.5,
        DOT_PRECISION=precision,
        TERM1_MASS_EXP=bool(term1_mass_exp),
        GROUP_SIZE=group_size,
        BLOCK_B=block_b,
        BLOCK_D=head_dim,
        num_warps=int(num_warps),
    )

    global_stats = None
    if reduce_globals:
        global_stats = _workspace_empty(
            workspace,
            "tc_selection_global",
            (4, rows),
            device=q.device,
            dtype=torch.float32,
        )
        _selection_reduce_kernel[(rows,)](
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
