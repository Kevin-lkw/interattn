import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from torch.profiler import ProfilerActivity, profile

from .common import load_model_and_tokenizer
from .methods import build_method, generate_with_method
from ..runner_utils import set_seed


DEFAULT_REPOS = ("THUDM/LongBench-v2", "zai-org/LongBench-v2")


@dataclass
class LatencyRow:
    method: str
    target_context_tokens: int
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    record_id: str
    record_length: str | None
    phase: str
    metadata: dict
    decode_only_seconds: float | None = None
    profile: dict | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Latency-only comparison on LongBench v2 long contexts. "
            "This script does not evaluate accuracy."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "condition_block_triton"],
        choices=[
            "full",
            "condition_block",
            "condition_block_triton",
            "condition_block_triton_term1_softmax",
        ],
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1, help="Profiled samples per method/context after warmup.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup samples per method/context; not included in summary.")
    parser.add_argument("--condition-block-size", type=int, default=32)
    parser.add_argument("--condition-eps", type=float, default=0.1)
    parser.add_argument("--full-attention-layers", type=int, default=0)
    parser.add_argument("--skip-stats", action="store_true", help="Set CONDITION_BLOCK_SKIP_STATS=1 for condition-block methods.")
    parser.add_argument("--compile-selection", action="store_true", help="Set CONDITION_BLOCK_COMPILE_SELECTION=1 for Triton condition-block.")
    parser.add_argument(
        "--mixed-summaries",
        action="store_true",
        help="Set CONDITION_BLOCK_MIXED_SUMMARIES=1 for the exact mixed summary layout.",
    )
    parser.add_argument(
        "--tma-bounds",
        action="store_true",
        help="Set CONDITION_BLOCK_TMA_BOUNDS=1 for packed Hopper/Blackwell bounds loads.",
    )
    parser.add_argument("--triton-chunk-blocks", type=int, default=64)
    parser.add_argument("--hf-repo", default=None, help="Override LongBench v2 HF repo. Default tries THUDM then zai-org.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="Collect torch profiler CUDA breakdown for measured samples.")
    parser.add_argument(
        "--profile-decode-only",
        action="store_true",
        help="With --profile and --decode-only-timing, start the profiler after the prefill forward returns.",
    )
    parser.add_argument(
        "--decode-only-timing",
        action="store_true",
        help=(
            "Also report wall-clock time after the first model forward returns. "
            "For generate-style runs, the first forward is the long-context prefill, "
            "so this approximates decode-only latency without changing the generate path."
        ),
    )
    parser.add_argument(
        "--fixed-decode",
        action="store_true",
        help="Force generation to run max_new_tokens instead of stopping at EOS; recommended for latency-only runs.",
    )
    parser.add_argument("--allow-over-model-max", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("llama/result/generate/longbench_v2_latency.jsonl"))
    return parser.parse_args()


def make_generation_args(args, method_name):
    return SimpleNamespace(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
        attn_implementation=None,
        method=method_name,
        budget=None,
        max_new_tokens=args.max_new_tokens,
        full_attention_layers=args.full_attention_layers,
        condition_eps=args.condition_eps,
        condition_block_size=args.condition_block_size,
        condition_delta_mode="range_bound",
        quest_page_size=None,
        kvpress_window_size=64,
        kvpress_kernel_size=5,
        kvpress_alpha_safeguard=0.20,
        kvpress_sink_tokens=4,
    )


def load_longbench_v2(repo, split):
    errors = []
    repos = [repo] if repo else list(DEFAULT_REPOS)
    for candidate in repos:
        try:
            return candidate, load_dataset(candidate, split=split, trust_remote_code=True)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Failed to load LongBench v2 from: " + " | ".join(errors))


def format_longbench_v2_prompt(record):
    choices = "\n".join(
        [
            f"A. {record.get('choice_A', '')}",
            f"B. {record.get('choice_B', '')}",
            f"C. {record.get('choice_C', '')}",
            f"D. {record.get('choice_D', '')}",
        ]
    )
    return (
        "Please answer the multiple-choice question using the provided long context.\n\n"
        f"Context:\n{record.get('context', '')}\n\n"
        f"Question: {record.get('question', '')}\n"
        f"{choices}\n\n"
        "Answer with only one option letter:"
    )


def middle_truncate_ids(input_ids, target_tokens):
    if int(input_ids.shape[1]) <= target_tokens:
        return input_ids
    left = target_tokens // 2
    right = target_tokens - left
    return torch.cat([input_ids[:, :left], input_ids[:, -right:]], dim=1)


def find_inputs_for_context(dataset, tokenizer, target_tokens, start_index, device):
    for idx in range(start_index, len(dataset)):
        record = dict(dataset[idx])
        prompt = format_longbench_v2_prompt(record)
        ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
        if int(ids.shape[1]) >= target_tokens:
            ids = middle_truncate_ids(ids, target_tokens).to(device)
            return idx, record, ids, torch.ones_like(ids, device=device)
    raise ValueError(f"No LongBench v2 record with at least {target_tokens} tokens after index {start_index}.")


def classify_kernel(name):
    s = name.lower()
    if "condition_block_finalize" in s:
        return "sparse_finalize_attention"
    if "condition_block_stage2_reduce" in s:
        return "sparse_reduce"
    if "condition_block_select" in s or "selection" in s:
        return "condition_selection"
    if "flash" in s or "scaled_dot_product" in s or "sdpa" in s:
        return "flash_sdpa_attention"
    if "gemm" in s or "gemv" in s or "aten::mm" in s or "aten::matmul" in s or "cublas" in s:
        return "model_gemm_gemv"
    if "cat" in s or "slice" in s or "index" in s or "copy" in s or "memcpy" in s:
        return "kv_cache_copy_index"
    if "softmax" in s:
        return "softmax"
    if "norm" in s or "silu" in s or "mul" in s or "add" in s or "rope" in s or "rotary" in s:
        return "pointwise_norm_rope"
    return "other"


def summarize_profiler(prof):
    cats = defaultdict(lambda: {"cuda_us": 0.0, "cpu_us": 0.0, "count": 0})
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_cuda_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(evt, "self_device_time_total", 0.0)
        cat = classify_kernel(evt.key)
        cats[cat]["cuda_us"] += float(cuda_us)
        cats[cat]["cpu_us"] += float(evt.self_cpu_time_total)
        cats[cat]["count"] += int(evt.count)
    total_cuda = sum(v["cuda_us"] for v in cats.values())
    return {
        "total_self_cuda_ms": total_cuda / 1000.0,
        "categories": {
            key: {
                "count": val["count"],
                "cuda_ms": val["cuda_us"] / 1000.0,
                "cuda_pct": 100.0 * val["cuda_us"] / total_cuda if total_cuda else 0.0,
                "cpu_ms": val["cpu_us"] / 1000.0,
            }
            for key, val in sorted(cats.items(), key=lambda item: item[1]["cuda_us"], reverse=True)
        },
    }


def run_generate(model, tokenizer, method, args, input_ids, attention_mask, decode_profiler=None):
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    orig_generation_eos = getattr(model.generation_config, "eos_token_id", None)
    orig_config_eos = getattr(model.config, "eos_token_id", None)
    orig_tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if args.fixed_decode:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        if method.kind in {
            "condition_block",
            "condition_block_triton",
            "condition_block_triton_term1_softmax",
        }:
            tokenizer.eos_token_id = None
    orig_forward = model.forward
    decode_timing = {"started": False, "start": None}

    if args.decode_only_timing:
        def timed_forward(*forward_args, **forward_kwargs):
            outputs = orig_forward(*forward_args, **forward_kwargs)
            if not decode_timing["started"]:
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                decode_timing["started"] = True
                decode_timing["start"] = time.perf_counter()
                if decode_profiler is not None:
                    decode_profiler.start()
            return outputs

        model.forward = timed_forward
    try:
        if args.fixed_decode and method.kind == "full":
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=method.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                )[:, input_ids.shape[1] :]
            result = output_ids
        else:
            result = generate_with_method(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                method=method,
                device=args.device,
                dataset="longbench_v2",
            )
    finally:
        model.forward = orig_forward
        if args.fixed_decode:
            model.generation_config.eos_token_id = orig_generation_eos
            model.config.eos_token_id = orig_config_eos
            tokenizer.eos_token_id = orig_tokenizer_eos
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    if decode_profiler is not None and decode_timing["started"]:
        decode_profiler.stop()
    seconds = time.perf_counter() - start
    output_ids, metadata = result if isinstance(result, tuple) else (result, {})
    decode_only_seconds = None
    if args.decode_only_timing and decode_timing["start"] is not None:
        decode_only_seconds = time.perf_counter() - float(decode_timing["start"])
        metadata = dict(metadata)
        metadata["decode_only_seconds"] = decode_only_seconds
    return output_ids, metadata, seconds, decode_only_seconds


def main():
    args = parse_args()
    set_seed(42)
    if args.skip_stats:
        os.environ["CONDITION_BLOCK_SKIP_STATS"] = "1"
    if args.compile_selection:
        os.environ["CONDITION_BLOCK_COMPILE_SELECTION"] = "1"
    if args.mixed_summaries:
        os.environ["CONDITION_BLOCK_MIXED_SUMMARIES"] = "1"
    if args.tma_bounds:
        os.environ["CONDITION_BLOCK_TMA_BOUNDS"] = "1"
    os.environ.setdefault("CONDITION_BLOCK_TRITON_CHUNK_BLOCKS", str(args.triton_chunk_blocks))

    repo, dataset = load_longbench_v2(args.hf_repo, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        gen_args = make_generation_args(args, method_name)
        model, tokenizer = load_model_and_tokenizer(gen_args)
        max_pos = getattr(model.config, "max_position_embeddings", None)
        method = build_method(gen_args)

        with args.output.open("a", encoding="utf-8") as handle:
            for target in args.contexts:
                effective_target = int(target)
                skipped_reason = None
                if max_pos is not None and not args.allow_over_model_max:
                    max_input = int(max_pos) - int(args.max_new_tokens)
                    if max_input <= 0:
                        skipped_reason = f"max_position_embeddings={max_pos} <= max_new_tokens={args.max_new_tokens}"
                    elif effective_target > max_input:
                        effective_target = max_input
                if skipped_reason is not None:
                    handle.write(json.dumps({
                        "method": method_name,
                        "target_context_tokens": int(target),
                        "skipped": True,
                        "reason": skipped_reason,
                    }) + "\n")
                    continue

                need = args.warmup + args.samples
                record_index = args.record_offset
                prepared = []
                for _ in range(need):
                    idx, record, input_ids, attention_mask = find_inputs_for_context(
                        dataset, tokenizer, effective_target, record_index, args.device
                    )
                    prepared.append((idx, record, input_ids, attention_mask))
                    record_index = idx + 1

                for phase_i, (idx, record, input_ids, attention_mask) in enumerate(prepared):
                    phase = "warmup" if phase_i < args.warmup else "measured"
                    prof_summary = None
                    if args.profile and phase == "measured":
                        if args.profile_decode_only:
                            if not args.decode_only_timing:
                                raise ValueError("--profile-decode-only requires --decode-only-timing.")
                            prof = profile(
                                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                record_shapes=False,
                                with_stack=False,
                            )
                            output_ids, metadata, seconds, decode_only_seconds = run_generate(
                                model,
                                tokenizer,
                                method,
                                args,
                                input_ids,
                                attention_mask,
                                decode_profiler=prof,
                            )
                            prof.step()
                            prof_summary = summarize_profiler(prof)
                        else:
                            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
                                output_ids, metadata, seconds, decode_only_seconds = run_generate(
                                    model, tokenizer, method, args, input_ids, attention_mask
                                )
                                prof.step()
                            prof_summary = summarize_profiler(prof)
                    else:
                        output_ids, metadata, seconds, decode_only_seconds = run_generate(
                            model, tokenizer, method, args, input_ids, attention_mask
                        )

                    row = LatencyRow(
                        method=method_name,
                        target_context_tokens=int(target),
                        input_tokens=int(input_ids.shape[1]),
                        output_tokens=int(output_ids.shape[1]),
                        generation_seconds=float(seconds),
                        record_id=str(record.get("_id", idx)),
                        record_length=record.get("length"),
                        phase=phase,
                        metadata=metadata,
                        decode_only_seconds=decode_only_seconds,
                        profile=prof_summary,
                    )
                    payload = asdict(row)
                    payload["hf_repo"] = repo
                    payload["model"] = args.model
                    payload["max_new_tokens"] = args.max_new_tokens
                    payload["condition_block_size"] = args.condition_block_size
                    payload["condition_eps"] = args.condition_eps
                    payload["mixed_summaries"] = args.mixed_summaries
                    payload["tma_bounds"] = args.tma_bounds
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
