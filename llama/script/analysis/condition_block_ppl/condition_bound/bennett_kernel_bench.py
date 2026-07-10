"""Microbench: Bennett-summary stats kernel vs production stats kernel.

Production loads 3 vectors/block (k_bar, k_max, k_min). The Bennett variant loads
2 vectors + 2 scalars (k_bar, diag key variance D, radius r, off-diag remainder
lambda), computes delta = ||q|| r scale (scalar radius), sigma^2 via D/lambda, and
G instead of cosh; it writes one extra cache row (sigma2). Cold-L2 methodology and
production launch configs as in selection_kernel_bench.py.
"""

import argparse

import torch
import triton
import triton.language as tl

from . import common  # noqa: F401

from analysis.condition_block_gen.methods.condition_block_triton_impl.core import (  # noqa: E402
    _condition_block_selection_stats_kernel,
    _select_warps,
)


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _stats_kernel_bennett(
    q_ptr, k_bar_ptr, var_diag_ptr, radius_ptr, lam_ptr, counts_ptr,
    s_cache_ptr, delta_cache_ptr, g_cache_ptr,
    z_m_ptr, z_l_ptr, c_m_ptr, c_l_ptr,
    n_blocks, n_chunks,
    group_size: tl.constexpr, head_dim: tl.constexpr, scale: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
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
    q = tl.load(q_ptr + row[:, None] * head_dim + d[None, :],
                mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    q_sq = q * q
    q_norm2 = tl.sum(q_sq, axis=1)
    count = tl.load(counts_ptr + b, mask=b_mask, other=0)
    active = b_mask & (count > 0)
    stat_off = ((kv_head * n_blocks + b[:, None]) * head_dim) + d[None, :]
    k_bar = tl.load(k_bar_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    var_d = tl.load(var_diag_ptr + stat_off, mask=active[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    radius = tl.load(radius_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)
    lam = tl.load(lam_ptr + kv_head * n_blocks + b, mask=active, other=0.0).to(tl.float32)

    s = tl.sum(q[:, None, :] * k_bar[None, :, :], axis=2) * scale
    sigma2 = (tl.sum(q_sq[:, None, :] * var_d[None, :, :], axis=2)
              + lam[None, :] * q_norm2[:, None]) * (scale * scale)
    delta = tl.sqrt(q_norm2)[:, None] * radius[None, :] * scale
    delta = tl.minimum(delta, 80.0)

    z = s + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    active_2d = g_mask[:, None] & active[None, :]
    z = tl.where(active_2d, z, -float("inf"))
    safe_d = tl.maximum(delta, 1e-20)
    G = (sigma2 * tl.exp(delta) + delta * delta * tl.exp(-sigma2 / safe_d)) \
        / tl.maximum(sigma2 + delta * delta, 1e-20)
    cosh_delta = 0.5 * (tl.exp(delta) + tl.exp(-delta))
    G = tl.minimum(tl.maximum(G, 1.0), cosh_delta)
    zc = z + tl.log(G)

    tl.store(s_cache_ptr + row[:, None] * n_blocks + b[None, :], s, mask=active_2d)
    tl.store(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], delta, mask=active_2d)
    tl.store(g_cache_ptr + row[:, None] * n_blocks + b[None, :], G, mask=active_2d)
    z_m = tl.max(z, axis=1)
    c_m = tl.max(zc, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    c_l = tl.sum(tl.exp(zc - c_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=g_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=g_mask)
    tl.store(c_m_ptr + partial_off, c_m, mask=g_mask)
    tl.store(c_l_ptr + partial_off, c_l, mask=g_mask)


def bench(n_blocks, n_sets=8, iters=200):
    torch.manual_seed(0)
    n_kv_heads, group_size, head_dim = 8, 4, 128
    rows = n_kv_heads * group_size
    chunk_b = 16
    n_chunks = triton.cdiv(n_blocks, chunk_b)
    warps = _select_warps(n_blocks)

    sets = []
    for _ in range(n_sets):
        q = torch.randn(rows, head_dim, device="cuda")
        k_bar = torch.randn(n_kv_heads, n_blocks, head_dim, device="cuda")
        spread = torch.rand(n_kv_heads, n_blocks, head_dim, device="cuda") * 0.5
        var_diag = spread ** 2
        radius = torch.rand(n_kv_heads, n_blocks, device="cuda") * 4.0
        lam = torch.rand(n_kv_heads, n_blocks, device="cuda") * 0.1
        counts = torch.full((n_blocks,), 32, device="cuda", dtype=torch.int32)
        sets.append(dict(
            q=q, k_bar=k_bar, k_max=k_bar + spread, k_min=k_bar - spread,
            var_diag=var_diag, radius=radius, lam=lam, counts=counts,
            s=torch.empty(rows, n_blocks, device="cuda"),
            dl=torch.empty(rows, n_blocks, device="cuda"),
            gc=torch.empty(rows, n_blocks, device="cuda"),
            p=torch.empty(4, rows, n_chunks, device="cuda"),
        ))

    launch = dict(group_size=group_size, head_dim=head_dim, scale=head_dim ** -0.5,
                  BLOCK_G=triton.next_power_of_2(group_size), BLOCK_B=chunk_b,
                  BLOCK_D=triton.next_power_of_2(head_dim))

    def run_base(t):
        _condition_block_selection_stats_kernel[(n_kv_heads, n_chunks)](
            t["q"], t["k_bar"], t["k_max"], t["k_min"], t["counts"], t["s"], t["dl"],
            t["p"][0], t["p"][1], t["p"][2], t["p"][3],
            n_blocks, n_chunks, num_warps=warps, **launch)

    def run_bennett(t):
        _stats_kernel_bennett[(n_kv_heads, n_chunks)](
            t["q"], t["k_bar"], t["var_diag"], t["radius"], t["lam"], t["counts"],
            t["s"], t["dl"], t["gc"],
            t["p"][0], t["p"][1], t["p"][2], t["p"][3],
            n_blocks, n_chunks, num_warps=warps, **launch)

    results = {}
    for name, fn in (("base(3vec)", run_base), ("bennett(2vec+2s)", run_bennett)):
        for t in sets:
            fn(t)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(iters):
            fn(sets[i % n_sets])
        end.record()
        torch.cuda.synchronize()
        results[name] = start.elapsed_time(end) / iters * 1000
    return results


def main():
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    print(f"device={torch.cuda.get_device_name()}")
    for n_blocks, ctx_name in ((1024, "32K"), (2048, "64K"), (4096, "128K")):
        r = bench(n_blocks)
        base, ben = r["base(3vec)"], r["bennett(2vec+2s)"]
        print(f"{ctx_name:5s} n_blocks={n_blocks:5d} warps={_select_warps(n_blocks)} | "
              f"base={base:8.2f}us  bennett={ben:8.2f}us  speedup={base / ben:.3f}x")


if __name__ == "__main__":
    main()
