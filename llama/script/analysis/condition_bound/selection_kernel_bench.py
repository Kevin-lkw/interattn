"""Microbench: cost of the S_delta normalizer (c_m/c_l accumulator) in the
selection-stats kernel. Cold-L2 methodology (rotate 8 input sets), production
shapes (Llama-3.1-8B decode: 8 kv heads, group 4, head_dim 128, block 32).

Result (RTX PRO 6000 Blackwell): removing the accumulator saves ~0.1% — the
kernel is read-bound on k_bar/k_max/k_min; the normalizer is effectively free.
"""

import argparse

import torch
import triton
import triton.language as tl

import common  # noqa: F401  (adds llama/script to sys.path)

from analysis.generate.methods.condition_block_triton_impl.core import (  # noqa: E402
    _condition_block_selection_stats_kernel,
    _select_warps,
)


@triton.jit(
    do_not_specialize=["n_blocks", "n_chunks"],
    do_not_specialize_on_alignment=["n_blocks", "n_chunks"],
)
def _stats_kernel_no_norm(
    q_ptr, k_bar_ptr, k_max_ptr, k_min_ptr, counts_ptr,
    s_cache_ptr, delta_cache_ptr, z_m_ptr, z_l_ptr,
    n_blocks, n_chunks,
    group_size: tl.constexpr, head_dim: tl.constexpr, scale: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Identical to _condition_block_selection_stats_kernel minus zc/c_m/c_l.
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
    tl.store(s_cache_ptr + row[:, None] * n_blocks + b[None, :], s, mask=active_2d)
    tl.store(delta_cache_ptr + row[:, None] * n_blocks + b[None, :], delta, mask=active_2d)
    z_m = tl.max(z, axis=1)
    z_l = tl.sum(tl.exp(z - z_m[:, None]), axis=1)
    partial_off = row * n_chunks + chunk
    tl.store(z_m_ptr + partial_off, z_m, mask=g_mask)
    tl.store(z_l_ptr + partial_off, z_l, mask=g_mask)


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
        counts = torch.full((n_blocks,), 32, device="cuda", dtype=torch.int32)
        sets.append((q, k_bar, k_bar + spread, k_bar - spread, counts,
                     torch.empty(rows, n_blocks, device="cuda"),
                     torch.empty(rows, n_blocks, device="cuda"),
                     torch.empty(4, rows, n_chunks, device="cuda")))

    launch = dict(group_size=group_size, head_dim=head_dim, scale=head_dim ** -0.5,
                  BLOCK_G=triton.next_power_of_2(group_size), BLOCK_B=chunk_b,
                  BLOCK_D=triton.next_power_of_2(head_dim))

    def run_base(st):
        q, kb, kh, kl, c, sc, dc, p = st
        _condition_block_selection_stats_kernel[(n_kv_heads, n_chunks)](
            q, kb, kh, kl, c, sc, dc, p[0], p[1], p[2], p[3],
            n_blocks, n_chunks, num_warps=warps, **launch)

    def run_nonorm(st):
        q, kb, kh, kl, c, sc, dc, p = st
        _stats_kernel_no_norm[(n_kv_heads, n_chunks)](
            q, kb, kh, kl, c, sc, dc, p[0], p[1],
            n_blocks, n_chunks, num_warps=warps, **launch)

    results = {}
    for name, fn in (("base(4-acc)", run_base), ("no-norm(2-acc)", run_nonorm)):
        for st in sets:
            fn(st)
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
        base, non = r["base(4-acc)"], r["no-norm(2-acc)"]
        print(f"{ctx_name:5s} n_blocks={n_blocks:5d} warps={_select_warps(n_blocks)} | "
              f"base={base:8.2f}us  no-norm={non:8.2f}us  speedup={base / non:.3f}x")


if __name__ == "__main__":
    main()
