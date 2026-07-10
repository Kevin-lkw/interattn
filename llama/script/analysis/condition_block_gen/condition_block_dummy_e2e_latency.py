import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

from .common import load_model_and_tokenizer
from .condition_block_stage_latency import (
    build_dummy_selected,
    run_sparse_attention_dummy,
)
from .longbench_v2_latency import (
    find_inputs_for_context,
    load_longbench_v2,
    summarize_profiler,
)
from .methods import build_method, generate_with_method
from .methods.condition_block_triton_impl import core as triton_core
from ..runner_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end condition_block_triton latency with dummy cluster selection. "
            "This keeps model forward/cache/hook/sparse-attention costs, but bypasses "
            "condition selection."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--condition-block-size", type=int, default=32)
    parser.add_argument("--condition-eps", type=float, default=0.1)
    parser.add_argument("--selected-ratios", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--triton-chunk-blocks", type=int, default=64)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument("--fixed-decode", action="store_true")
    parser.add_argument("--include-production", action="store_true")
    parser.add_argument("--decode-only-timing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-decode-only",
        action="store_true",
        help="With --profile and --decode-only-timing, start profiler after prefill returns.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/condition_block_dummy_e2e_latency.jsonl"))
    return parser.parse_args()


def make_generation_args(args):
    return SimpleNamespace(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
        attn_implementation=None,
        method="condition_block_triton",
        budget=None,
        max_new_tokens=args.max_new_tokens,
        full_attention_layers=0,
        condition_eps=args.condition_eps,
        condition_block_size=args.condition_block_size,
        condition_delta_mode="range_bound",
        quest_page_size=None,
        kvpress_window_size=64,
        kvpress_kernel_size=5,
        kvpress_alpha_safeguard=0.20,
        kvpress_sink_tokens=4,
    )


def make_dummy_fused_attention(selected_ratio):
    def _dummy_fused_attention(
        *,
        q_grouped,
        prompt_prefix,
        k_suffix,
        v_suffix,
        eps,
        page_size,
        store_selected,
        output_dtype,
        workspace=None,
    ):
        n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
        if n_query != 1:
            raise ValueError("dummy e2e fused attention expects q_len=1")
        n_blocks = int(prompt_prefix["block_valid_counts"].numel())
        selected = build_dummy_selected(
            n_kv_heads=n_kv_heads,
            n_blocks=n_blocks,
            ratio=selected_ratio,
            device=q_grouped.device,
        )
        q = q_grouped.reshape(n_kv_heads * group_size, head_dim).contiguous()
        out = run_sparse_attention_dummy(
            q=q,
            prefix=prompt_prefix,
            selected=selected,
            k_suffix=k_suffix,
            v_suffix=v_suffix,
            group_size=group_size,
            head_dim=head_dim,
            block_size=int(page_size),
            output_dtype=output_dtype,
        )
        selected_expanded = None
        if store_selected:
            selected_expanded = selected[:, None, None, :].expand(
                n_kv_heads,
                group_size,
                1,
                n_blocks,
            )
        return out.reshape(n_kv_heads, group_size, 1, head_dim), selected_expanded

    return _dummy_fused_attention


def run_generate_timed(model, tokenizer, method, args, input_ids, attention_mask, decode_profiler=None):
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    orig_generation_eos = getattr(model.generation_config, "eos_token_id", None)
    orig_config_eos = getattr(model.config, "eos_token_id", None)
    orig_tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if args.fixed_decode:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
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
    start = time.perf_counter()
    try:
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
    os.environ["CONDITION_BLOCK_SKIP_STATS"] = "1"
    os.environ["CONDITION_BLOCK_COMPILE_SELECTION"] = "1"
    os.environ.setdefault("CONDITION_BLOCK_TRITON_CHUNK_BLOCKS", str(args.triton_chunk_blocks))
    os.environ.setdefault("CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE", "1")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    repo, dataset = load_longbench_v2(args.hf_repo, args.split)
    gen_args = make_generation_args(args)
    model, tokenizer = load_model_and_tokenizer(gen_args)
    method = build_method(gen_args)

    original_fused = triton_core._condition_block_decode_output_fused_triton
    configs = []
    if args.include_production:
        configs.append(("production", None))
    configs.extend((f"dummy_ratio_{ratio:g}", float(ratio)) for ratio in args.selected_ratios)

    with args.output.open("w", encoding="utf-8") as handle:
        for context in args.contexts:
            start_index = int(args.record_offset)
            inputs = []
            total_needed = int(args.warmup + args.samples)
            for _ in range(total_needed):
                idx, record, input_ids, attention_mask = find_inputs_for_context(
                    dataset,
                    tokenizer,
                    int(context),
                    start_index,
                    args.device,
                )
                inputs.append((idx, record, input_ids, attention_mask))
                start_index = idx + 1

            for label, ratio in configs:
                if ratio is None:
                    triton_core._condition_block_decode_output_fused_triton = original_fused
                else:
                    triton_core._condition_block_decode_output_fused_triton = make_dummy_fused_attention(ratio)
                try:
                    for phase_idx, (_idx, record, input_ids, attention_mask) in enumerate(inputs):
                        phase = "warmup" if phase_idx < args.warmup else "measured"
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
                                output_ids, metadata, seconds, decode_only_seconds = run_generate_timed(
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
                                with profile(
                                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    record_shapes=False,
                                    with_stack=False,
                                ) as prof:
                                    output_ids, metadata, seconds, decode_only_seconds = run_generate_timed(
                                        model,
                                        tokenizer,
                                        method,
                                        args,
                                        input_ids,
                                        attention_mask,
                                    )
                                    prof.step()
                                prof_summary = summarize_profiler(prof)
                        else:
                            output_ids, metadata, seconds, decode_only_seconds = run_generate_timed(
                                model,
                                tokenizer,
                                method,
                                args,
                                input_ids,
                                attention_mask,
                            )
                        row = {
                            "config": label,
                            "dummy_selected_ratio": ratio,
                            "target_context_tokens": int(context),
                            "input_tokens": int(input_ids.shape[1]),
                            "output_tokens": int(output_ids.shape[-1]),
                            "generation_seconds": seconds,
                            "decode_only_seconds": decode_only_seconds,
                            "phase": phase,
                            "record_id": record.get("_id") or record.get("id"),
                            "record_length": record.get("length"),
                            "metadata": metadata,
                            "hf_repo": repo,
                            "model": args.model,
                            "max_new_tokens": int(args.max_new_tokens),
                            "condition_block_size": int(args.condition_block_size),
                            "profile": prof_summary,
                        }
                        print(json.dumps(row), flush=True)
                        handle.write(json.dumps(row) + "\n")
                finally:
                    triton_core._condition_block_decode_output_fused_triton = original_fused


if __name__ == "__main__":
    main()
