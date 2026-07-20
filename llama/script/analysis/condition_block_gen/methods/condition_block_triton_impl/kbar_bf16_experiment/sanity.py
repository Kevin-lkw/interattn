"""Compare FP32 and BF16 k_bar storage with otherwise identical summaries."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager

import torch

from ..core import (
    _build_prompt_blocks,
    _condition_block_decode_output_fused_triton,
    _run_condition_block_selection_stats,
    _select_prompt_blocks_triton,
)


@contextmanager
def _summary_layout(*, bf16_k_bar, tma_bounds):
    updates = {
        "CONDITION_BLOCK_MIXED_SUMMARIES": "1",
        "CONDITION_BLOCK_K_BAR_DTYPE": "bfloat16" if bf16_k_bar else "float32",
        "CONDITION_BLOCK_TMA_BOUNDS": "1" if tma_bounds else None,
    }
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_prefix(k_prompt, v_prompt, block_size, *, bf16_k_bar, tma_bounds):
    with _summary_layout(bf16_k_bar=bf16_k_bar, tma_bounds=tma_bounds):
        return _build_prompt_blocks(k_prompt, v_prompt, block_size)


def _evaluate(q_grouped, prefix, k_suffix, v_suffix, eps, block_size, tma_bounds):
    with _summary_layout(bf16_k_bar=prefix["k_bar"].dtype == torch.bfloat16, tma_bounds=tma_bounds):
        _q, s, delta, _partial, global_stats, _n_blocks, _n_chunks = (
            _run_condition_block_selection_stats(q_grouped, prefix)
        )
        selected, _z, _v_bar, _size, _exists = _select_prompt_blocks_triton(
            q_grouped, prefix, eps
        )
        suffix_len = torch.tensor(
            k_suffix.shape[1], device=q_grouped.device, dtype=torch.int32
        )
        output, fused_selected = _condition_block_decode_output_fused_triton(
            q_grouped=q_grouped,
            prompt_prefix=prefix,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            suffix_len_dev=suffix_len,
            eps=eps,
            page_size=block_size,
            store_selected=True,
            output_dtype=torch.bfloat16,
            workspace={},
        )
        torch.cuda.synchronize(q_grouped.device)
    return {
        "s": s.clone(),
        "delta": delta.clone(),
        "global": global_stats.clone(),
        "selected": selected.clone(),
        "fused_selected": fused_selected.clone(),
        "output": output.clone(),
    }


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def run_case(*, device, block_size, prompt_len, suffix_len, eps, tma_bounds):
    kv_heads, group_size, head_dim = 8, 4, 128
    k_prompt = torch.randn(
        (kv_heads, prompt_len, head_dim), device=device, dtype=torch.bfloat16
    )
    v_prompt = torch.randn_like(k_prompt)
    q_grouped = torch.randn(
        (kv_heads, group_size, 1, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    k_suffix = torch.randn(
        (kv_heads, suffix_len, head_dim), device=device, dtype=torch.bfloat16
    )
    v_suffix = torch.randn_like(k_suffix)

    baseline_prefix = _build_prefix(
        k_prompt, v_prompt, block_size, bf16_k_bar=False, tma_bounds=tma_bounds
    )
    candidate_prefix = _build_prefix(
        k_prompt, v_prompt, block_size, bf16_k_bar=True, tma_bounds=tma_bounds
    )
    baseline = _evaluate(
        q_grouped,
        baseline_prefix,
        k_suffix,
        v_suffix,
        eps,
        block_size,
        tma_bounds,
    )
    candidate = _evaluate(
        q_grouped,
        candidate_prefix,
        k_suffix,
        v_suffix,
        eps,
        block_size,
        tma_bounds,
    )
    return {
        "block_size": block_size,
        "prompt_len": prompt_len,
        "suffix_len": suffix_len,
        "eps": eps,
        "tma_bounds": tma_bounds,
        "baseline_k_bar_dtype": str(baseline_prefix["k_bar"].dtype),
        "candidate_k_bar_dtype": str(candidate_prefix["k_bar"].dtype),
        "s_max_abs": _max_abs(baseline["s"], candidate["s"]),
        "delta_max_abs": _max_abs(baseline["delta"], candidate["delta"]),
        "global_max_abs": _max_abs(baseline["global"], candidate["global"]),
        "selected_mismatch": int(
            (baseline["selected"] != candidate["selected"]).sum().item()
        ),
        "fused_selected_mismatch": int(
            (baseline["fused_selected"] != candidate["fused_selected"]).sum().item()
        ),
        "selected_total": int(baseline["selected"].numel()),
        "output_max_abs": _max_abs(baseline["output"], candidate["output"]),
        "output_mean_abs": float(
            (baseline["output"].float() - candidate["output"].float())
            .abs()
            .mean()
            .item()
        ),
        "output_exact": bool(torch.equal(baseline["output"], candidate["output"])),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = (
        (32, 4099, 33, 0.1, False),
        (64, 4099, 33, 0.1, False),
        (32, 4099, 33, 0.1, True),
        (64, 4099, 33, 0.1, True),
    )
    for block_size, prompt_len, suffix_len, eps, tma_bounds in cases:
        print(
            json.dumps(
                run_case(
                    device=device,
                    block_size=block_size,
                    prompt_len=prompt_len,
                    suffix_len=suffix_len,
                    eps=eps,
                    tma_bounds=tma_bounds,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
