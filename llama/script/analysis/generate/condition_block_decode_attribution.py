"""Kernel-level decode-only attribution for condition-block Triton vs full.

The bucket percentages from ``longbench_v2_latency.py --profile`` tell which
category dominates but not three things needed to pick the next optimization:

1. GPU busy vs idle: how much decode wall-clock is covered by kernels at all.
   A large idle share means the path is host/launch-bound and kernel work is
   not the right target.
2. Kernel-level detail: which individual kernels fill each bucket (e.g. what
   exactly is inside ``kv_cache_copy_index``), and how many kernel launches
   run per decode step.
3. Production sparse-path split: per-kernel time of the four condition-block
   kernels under real shapes and real selected ratios.

Per method/context this runs one warmup generate, one clean run for decode-only
wall time, and one profiled run for kernel times. Kernel durations are stable
under the profiler even though wall time inflates, so busy/wall combines the
profiled kernel total with the clean wall time.

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/scratch1/liankewei/interattn \
    CONDITION_BLOCK_SKIP_STATS=1 \
    CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE=1 \
    python -m llama.script.analysis.generate.condition_block_decode_attribution \
      --device cuda:0 --methods condition_block_triton \
      --contexts 32768 65536 131072 --max-new-tokens 128
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from .common import load_model_and_tokenizer
from .methods import build_method
from .longbench_v2_latency import (
    classify_kernel,
    find_inputs_for_context,
    load_longbench_v2,
    make_generation_args,
    run_generate,
)
from ..runner_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kernel-level decode-only attribution on LongBench v2 contexts."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "condition_block_triton"],
        choices=["full", "full_static", "condition_block_triton"],
        help="full_static runs the full method with HF StaticCache (cache_implementation='static').",
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--condition-block-size", type=int, default=32)
    parser.add_argument("--condition-eps", type=float, default=0.1)
    parser.add_argument("--top-kernels", type=int, default=40)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llama/result/generate/condition_block_decode_attribution.jsonl"),
    )
    return parser.parse_args()


def make_args_namespace(args, method_name):
    base_method = "full" if method_name == "full_static" else method_name
    gen_args = make_generation_args(
        SimpleNamespace(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            full_attention_layers=0,
            condition_eps=args.condition_eps,
            condition_block_size=args.condition_block_size,
        ),
        base_method,
    )
    run_args = SimpleNamespace(
        device=args.device,
        fixed_decode=True,
        decode_only_timing=True,
    )
    return gen_args, run_args


def summarize_kernel_events(prof, decode_steps, top_kernels):
    """Aggregate device-side kernel entries only (no CPU-op double counting)."""
    kernels = [
        evt
        for evt in prof.key_averages()
        if evt.device_type == DeviceType.CUDA and evt.self_device_time_total > 0
    ]
    total_us = sum(evt.self_device_time_total for evt in kernels)
    launches = sum(evt.count for evt in kernels)

    buckets = {}
    for evt in kernels:
        cat = classify_kernel(evt.key)
        entry = buckets.setdefault(cat, {"cuda_us": 0.0, "count": 0})
        entry["cuda_us"] += float(evt.self_device_time_total)
        entry["count"] += int(evt.count)

    top = sorted(kernels, key=lambda evt: evt.self_device_time_total, reverse=True)
    top_rows = [
        {
            "kernel": evt.key,
            "category": classify_kernel(evt.key),
            "count": int(evt.count),
            "cuda_ms": float(evt.self_device_time_total) / 1000.0,
            "cuda_pct": 100.0 * float(evt.self_device_time_total) / total_us if total_us else 0.0,
            "avg_us": float(evt.self_device_time_total) / max(int(evt.count), 1),
        }
        for evt in top[:top_kernels]
    ]
    return {
        "gpu_busy_ms": total_us / 1000.0,
        "kernel_launches": int(launches),
        "kernel_launches_per_step": launches / max(decode_steps, 1),
        "categories": {
            key: {
                "count": val["count"],
                "cuda_ms": val["cuda_us"] / 1000.0,
                "cuda_pct": 100.0 * val["cuda_us"] / total_us if total_us else 0.0,
            }
            for key, val in sorted(buckets.items(), key=lambda item: item[1]["cuda_us"], reverse=True)
        },
        "top_kernels": top_rows,
    }


def main():
    args = parse_args()
    set_seed(42)
    repo, dataset = load_longbench_v2(args.hf_repo, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        gen_args, run_args = make_args_namespace(args, method_name)
        model, tokenizer = load_model_and_tokenizer(gen_args)
        method = build_method(gen_args)
        orig_cache_impl = getattr(model.generation_config, "cache_implementation", None)
        if method_name == "full_static":
            model.generation_config.cache_implementation = "static"

        max_pos = getattr(model.config, "max_position_embeddings", None)
        with args.output.open("a", encoding="utf-8") as handle:
            for target in args.contexts:
                effective_target = int(target)
                if max_pos is not None:
                    effective_target = min(effective_target, int(max_pos) - args.max_new_tokens)
                idx, record, input_ids, attention_mask = find_inputs_for_context(
                    dataset, tokenizer, effective_target, args.record_offset, args.device
                )

                # Warmup (Triton JIT, allocator pools), then a clean run for
                # trustworthy decode-only wall time.
                run_generate(model, tokenizer, method, run_args, input_ids, attention_mask)
                output_ids, metadata, seconds, clean_decode_seconds = run_generate(
                    model, tokenizer, method, run_args, input_ids, attention_mask
                )
                decode_steps = int(output_ids.shape[1]) - 1

                prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                )
                run_generate(
                    model, tokenizer, method, run_args, input_ids, attention_mask,
                    decode_profiler=prof,
                )
                prof.step()
                summary = summarize_kernel_events(prof, decode_steps, args.top_kernels)
                del prof

                decode_ms = clean_decode_seconds * 1000.0
                row = {
                    "method": method_name,
                    "target_context_tokens": int(target),
                    "input_tokens": int(input_ids.shape[1]),
                    "decode_steps": decode_steps,
                    "decode_only_seconds": clean_decode_seconds,
                    "decode_ms_per_step": decode_ms / max(decode_steps, 1),
                    "gpu_busy_ms_per_step": summary["gpu_busy_ms"] / max(decode_steps, 1),
                    "gpu_busy_fraction": summary["gpu_busy_ms"] / decode_ms if decode_ms else None,
                    "model": args.model,
                    "hf_repo": repo,
                    "record_id": str(record.get("_id", idx)),
                    "condition_block_size": args.condition_block_size,
                    "condition_eps": args.condition_eps,
                    **summary,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{method_name} @ {target}] decode {row['decode_ms_per_step']:.2f} ms/step, "
                    f"GPU busy {row['gpu_busy_ms_per_step']:.2f} ms/step "
                    f"({100.0 * row['gpu_busy_fraction']:.1f}%), "
                    f"{row['kernel_launches_per_step']:.0f} launches/step",
                    flush=True,
                )
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

        if method_name == "full_static":
            model.generation_config.cache_implementation = orig_cache_impl
        del model
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
