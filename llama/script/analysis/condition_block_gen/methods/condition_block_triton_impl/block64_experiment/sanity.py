"""Compare the native block-64 fused kernel with the dense reference."""

from __future__ import annotations

import argparse
import json

import torch

from ..core import (
    _build_prompt_blocks,
    _condition_block_decode_output_fused_triton,
    _select_prompt_blocks_triton,
)
from ..legacy import _condition_block_decode_output_dense


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def run_case(*, device, prompt_len, suffix_len, eps):
    kv_heads = 8
    group_size = 4
    head_dim = 128
    block_size = 64
    k_prompt = torch.randn(
        (kv_heads, prompt_len, head_dim), device=device, dtype=torch.bfloat16
    )
    v_prompt = torch.randn_like(k_prompt)
    prefix = _build_prompt_blocks(k_prompt, v_prompt, block_size)
    q_grouped = torch.randn(
        (kv_heads, group_size, 1, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    k_suffix = torch.randn(
        (kv_heads, suffix_len, head_dim), device=device, dtype=torch.bfloat16
    )
    v_suffix = torch.randn_like(k_suffix)

    selected_ref, z_logits, v_bar, _size, cluster_exists = (
        _select_prompt_blocks_triton(q_grouped, prefix, eps)
    )
    pos_tensor = torch.tensor(
        [prompt_len + suffix_len - 1], device=device, dtype=torch.long
    )
    output_ref = _condition_block_decode_output_dense(
        q_grouped=q_grouped.float(),
        pos_tensor=pos_tensor,
        prompt_prefix=prefix,
        k_suffix=k_suffix.float(),
        v_suffix=v_suffix.float(),
        block_size=block_size,
        prompt_len=prompt_len,
        selected=selected_ref,
        z_logits=z_logits,
        v_bar=v_bar,
        cluster_exists=cluster_exists,
    ).to(torch.bfloat16)
    suffix_len_dev = torch.tensor(suffix_len, device=device, dtype=torch.int32)
    output, selected = _condition_block_decode_output_fused_triton(
        q_grouped=q_grouped,
        prompt_prefix=prefix,
        k_suffix=k_suffix,
        v_suffix=v_suffix,
        suffix_len_dev=suffix_len_dev,
        eps=eps,
        page_size=block_size,
        store_selected=True,
        output_dtype=torch.bfloat16,
        workspace={},
    )
    torch.cuda.synchronize(device)
    diff = (output.float() - output_ref.float()).abs()
    return {
        "prompt_len": prompt_len,
        "suffix_len": suffix_len,
        "eps": eps,
        "selected": int(selected_ref.sum().item()),
        "selected_total": int(selected_ref.numel()),
        "selected_mismatch": int((selected != selected_ref).sum().item()),
        "output_max_abs": float(diff.max().item()),
        "output_mean_abs": float(diff.mean().item()),
        "output_exact": bool(torch.equal(output, output_ref)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = (
        {"prompt_len": 4096, "suffix_len": 1, "eps": 0.0},
        {"prompt_len": 4099, "suffix_len": 33, "eps": 0.0},
        {"prompt_len": 4096, "suffix_len": 1, "eps": 1e6},
        {"prompt_len": 4099, "suffix_len": 33, "eps": 1e6},
    )
    for case in cases:
        print(json.dumps(run_case(device=device, **case)), flush=True)


if __name__ == "__main__":
    main()
