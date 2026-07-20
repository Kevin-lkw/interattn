"""Capture real LongBench tensors and replay full/sparse attention kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ..common import load_model_and_tokenizer
from ..longbench_v2_latency import (
    find_inputs_for_context,
    load_longbench_v2,
)
from ..methods.condition_block_triton_impl.core import (
    _condition_block_decode_output_fused_triton,
)
from ...runner_utils import set_seed
from .capture import LayerCapture, capture_first_decode_step
from .fixed_mask import fixed_mask_hybrid_attention


DEFAULT_OUTPUT_DIR = Path(
    "llama/result/generate/condition_block_real_attention_benchmark"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure full, production sparse, and fixed-real-mask attention "
            "on tensors captured from actual LongBench-v2 prompts."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--eps", nargs="+", type=float, default=[0.1])
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to capture; default captures every decoder layer.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _model_args(args):
    return SimpleNamespace(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
        attn_implementation=None,
        method="condition_block_triton",
    )


def _set_runtime_environment():
    os.environ["CONDITION_BLOCK_MIXED_SUMMARIES"] = "1"
    os.environ["CONDITION_BLOCK_K_BAR_DTYPE"] = "bfloat16"
    os.environ["CONDITION_BLOCK_SKIP_STATS"] = "1"
    for name in (
        "CONDITION_BLOCK_CUDA_GRAPH",
        "CONDITION_BLOCK_TMA_BOUNDS",
        "CONDITION_BLOCK_DENSE_STAGE2",
        "CONDITION_BLOCK_COMPACT_SDPA_STAGE2",
        "CONDITION_BLOCK_LEGACY_STAGE2",
        "CONDITION_BLOCK_EAGER_SELECTION",
    ):
        os.environ.pop(name, None)


def _cuda_time_ms(fn, *, warmup, iters, l2_flush):
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush.add_(1)
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        if l2_flush is not None:
            l2_flush.add_(1)
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def _full_sdpa(capture: LayerCapture, k_all, v_all):
    n_kv_heads, group_size, _, head_dim = capture.q_grouped.shape
    query = capture.q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
    return F.scaled_dot_product_attention(
        query,
        k_all.unsqueeze(0),
        v_all.unsqueeze(0),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=head_dim**-0.5,
        enable_gqa=group_size != 1,
    )


def _routing_metrics(capture: LayerCapture):
    selected = capture.selected
    counts = capture.prompt_prefix["block_valid_counts"]
    active = counts > 0
    selected_tokens = (selected.long() * counts.view(1, -1)).sum(dim=1).float()
    selected_blocks = selected.sum(dim=1).float()
    representative_blocks = ((~selected) & active.view(1, -1)).sum(dim=1).float()
    selected_tokens_mean = float(selected_tokens.mean().item())
    selected_blocks_mean = float(selected_blocks.mean().item())
    representatives_mean = float(representative_blocks.mean().item())
    suffix = capture.suffix_tokens
    prompt = capture.prompt_tokens

    # Candidate-count ideal: assumes identical cost per exact token and per
    # representative, and ignores all routing/launch/softmax overhead.
    full_candidates = float(prompt + suffix)
    hybrid_candidates = selected_tokens_mean + representatives_mean + suffix
    candidate_theoretical_speedup = full_candidates / hybrid_candidates

    # Optimistic IO model for the current mixed/BF16 block64 formulation:
    # selection reads k_bar/k_max/k_min (3 vectors/block), attention reads one
    # v_bar per unselected block, K+V for selected tokens, and exact suffix K+V.
    # Counts, norms, partial buffers and output writes are deliberately omitted.
    active_blocks = float(active.sum().item())
    full_vector_reads = 2.0 * (prompt + suffix)
    sparse_vector_reads = (
        3.0 * active_blocks
        + representatives_mean
        + 2.0 * selected_tokens_mean
        + 2.0 * suffix
    )
    io_theoretical_speedup = full_vector_reads / sparse_vector_reads
    return {
        "active_blocks": int(active_blocks),
        "selected_blocks_mean": selected_blocks_mean,
        "selected_block_ratio": selected_blocks_mean / max(active_blocks, 1.0),
        "selected_blocks_min": int(selected_blocks.min().item()),
        "selected_blocks_max": int(selected_blocks.max().item()),
        "selected_tokens_mean": selected_tokens_mean,
        "representative_blocks_mean": representatives_mean,
        "hybrid_candidates": hybrid_candidates,
        "candidate_theoretical_speedup": candidate_theoretical_speedup,
        "full_vector_reads": full_vector_reads,
        "sparse_vector_reads": sparse_vector_reads,
        "io_theoretical_speedup": io_theoretical_speedup,
    }


def _benchmark_capture(capture, *, block_size, eps, args, l2_flush):
    suffix_len_dev = torch.tensor(
        capture.suffix_tokens,
        device=capture.q_grouped.device,
        dtype=torch.int32,
    )
    production_workspace = {}
    production_output, fused_selected = _condition_block_decode_output_fused_triton(
        q_grouped=capture.q_grouped,
        prompt_prefix=capture.prompt_prefix,
        k_suffix=capture.k_suffix,
        v_suffix=capture.v_suffix,
        suffix_len_dev=suffix_len_dev,
        eps=float(eps),
        page_size=int(block_size),
        store_selected=True,
        output_dtype=capture.q_grouped.dtype,
        workspace=production_workspace,
    )
    expected_selected = capture.selected[:, None, None].expand_as(fused_selected)
    selected_exact = bool(torch.equal(fused_selected, expected_selected))

    fixed_workspace = {}
    fixed_output = fixed_mask_hybrid_attention(
        q_grouped=capture.q_grouped,
        prompt_prefix=capture.prompt_prefix,
        selected=capture.selected,
        k_suffix=capture.k_suffix,
        v_suffix=capture.v_suffix,
        block_size=block_size,
        output_dtype=capture.q_grouped.dtype,
        workspace=fixed_workspace,
    )
    torch.cuda.synchronize()
    fixed_max_abs = float(
        (production_output.float() - fixed_output.float()).abs().max().item()
    )
    if not selected_exact:
        raise AssertionError(f"Fused routing mismatch in layer {capture.layer_idx}")
    if fixed_max_abs > 2e-3:
        raise AssertionError(
            f"Fixed-mask output mismatch in layer {capture.layer_idx}: {fixed_max_abs}"
        )

    # Materialize an ideal contiguous full-attention baseline once, outside
    # the measured region. This prevents a strided StaticCache view copy from
    # being charged to every SDPA call.
    k_full = capture.k_all.contiguous()
    v_full = capture.v_all.contiguous()
    full_ms = _cuda_time_ms(
        lambda: _full_sdpa(capture, k_full, v_full),
        warmup=args.warmup,
        iters=args.iters,
        l2_flush=l2_flush,
    )
    production_ms = _cuda_time_ms(
        lambda: _condition_block_decode_output_fused_triton(
            q_grouped=capture.q_grouped,
            prompt_prefix=capture.prompt_prefix,
            k_suffix=capture.k_suffix,
            v_suffix=capture.v_suffix,
            suffix_len_dev=suffix_len_dev,
            eps=float(eps),
            page_size=int(block_size),
            store_selected=False,
            output_dtype=capture.q_grouped.dtype,
            workspace=production_workspace,
        ),
        warmup=args.warmup,
        iters=args.iters,
        l2_flush=l2_flush,
    )
    fixed_ms = _cuda_time_ms(
        lambda: fixed_mask_hybrid_attention(
            q_grouped=capture.q_grouped,
            prompt_prefix=capture.prompt_prefix,
            selected=capture.selected,
            k_suffix=capture.k_suffix,
            v_suffix=capture.v_suffix,
            block_size=block_size,
            output_dtype=capture.q_grouped.dtype,
            workspace=fixed_workspace,
        ),
        warmup=args.warmup,
        iters=args.iters,
        l2_flush=l2_flush,
    )
    routing = _routing_metrics(capture)
    production_speedup = full_ms / production_ms
    fixed_speedup = full_ms / fixed_ms
    return {
        **routing,
        "full_attention_ms": full_ms,
        "production_attention_ms": production_ms,
        "fixed_mask_attention_ms": fixed_ms,
        "production_speedup": production_speedup,
        "fixed_mask_speedup": fixed_speedup,
        "production_io_bound_realized": (
            production_speedup / routing["io_theoretical_speedup"]
        ),
        "fixed_candidate_bound_realized": (
            fixed_speedup / routing["candidate_theoretical_speedup"]
        ),
        "production_vs_fixed_ceiling": production_speedup / fixed_speedup,
        "fused_selected_exact": selected_exact,
        "fixed_output_max_abs": fixed_max_abs,
    }


def _prepare_inputs(dataset, tokenizer, args, max_position):
    prepared = []
    for target in args.contexts:
        effective = int(target)
        if max_position:
            effective = min(effective, max_position - 2)
        start = int(args.record_offset)
        for sample_idx in range(args.samples):
            index, record, input_ids, attention_mask = find_inputs_for_context(
                dataset,
                tokenizer,
                effective,
                start_index=start,
                device="cpu",
            )
            prepared.append(
                {
                    "target_context_tokens": int(target),
                    "effective_context_tokens": effective,
                    "sample_idx": sample_idx,
                    "record_index": int(index),
                    "record_id": str(record.get("_id", record.get("id", index))),
                    "record_length": record.get("length"),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            start = index + 1
    return prepared


def _summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["target_context_tokens"], row["block_size"], row["eps"])].append(row)
    summaries = []
    for (context, block_size, eps), group in sorted(groups.items()):
        full = sum(row["full_attention_ms"] for row in group)
        production = sum(row["production_attention_ms"] for row in group)
        fixed = sum(row["fixed_mask_attention_ms"] for row in group)
        full_vectors = sum(row["full_vector_reads"] for row in group)
        sparse_vectors = sum(row["sparse_vector_reads"] for row in group)
        hybrid_candidates = sum(row["hybrid_candidates"] for row in group)
        full_candidates = sum(
            row["input_tokens"] + row["suffix_tokens"] for row in group
        )
        production_speedup = full / production
        fixed_speedup = full / fixed
        io_bound = full_vectors / sparse_vectors
        candidate_bound = full_candidates / hybrid_candidates
        summaries.append(
            {
                "target_context_tokens": context,
                "block_size": block_size,
                "eps": eps,
                "captures": len(group),
                "samples": len({row["sample_key"] for row in group}),
                "mean_selected_block_ratio": statistics.mean(
                    row["selected_block_ratio"] for row in group
                ),
                "mean_full_attention_ms": full / len(group),
                "mean_production_attention_ms": production / len(group),
                "mean_fixed_mask_attention_ms": fixed / len(group),
                "production_speedup": production_speedup,
                "fixed_mask_speedup": fixed_speedup,
                "io_theoretical_speedup": io_bound,
                "candidate_theoretical_speedup": candidate_bound,
                "production_io_bound_realized": production_speedup / io_bound,
                "fixed_candidate_bound_realized": fixed_speedup / candidate_bound,
                "production_vs_fixed_ceiling": production_speedup / fixed_speedup,
                "max_fixed_output_abs": max(row["fixed_output_max_abs"] for row in group),
                "routing_exact": all(row["fused_selected_exact"] for row in group),
            }
        )
    return summaries


def _write_markdown(path, summaries, args, repo):
    lines = [
        "# Real LongBench attention capture benchmark",
        "",
        f"Source: `{repo}`. Each row uses real model Q/K/V and real routing from "
        "the first sparse decode step after SDPA prefill.",
        "",
        "Latency is cold-L2 device time for one decoder layer. It excludes prefill, "
        "model GEMV, cache updates and Python dispatch.",
        "",
        "| context | block | eps | samples/layers | selected blocks | full | production | fixed real mask | production speedup | fixed ceiling | IO ideal | IO ideal realized | runnable ceiling realized |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target_context_tokens']} | {row['block_size']} | {row['eps']:g} | "
            f"{row['samples']}/{row['captures']} | "
            f"{100 * row['mean_selected_block_ratio']:.2f}% | "
            f"{1000 * row['mean_full_attention_ms']:.2f} us | "
            f"{1000 * row['mean_production_attention_ms']:.2f} us | "
            f"{1000 * row['mean_fixed_mask_attention_ms']:.2f} us | "
            f"{row['production_speedup']:.2f}x | {row['fixed_mask_speedup']:.2f}x | "
            f"{row['io_theoretical_speedup']:.2f}x | "
            f"{100 * row['production_io_bound_realized']:.1f}% | "
            f"{100 * row['production_vs_fixed_ceiling']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The IO ideal counts only summary/page/suffix vector reads. It excludes "
            "norm/count/partial traffic, arithmetic, softmax/reduction and launch latency, "
            "so it is intentionally optimistic. The fixed-real-mask kernel is the more "
            "actionable executable ceiling.",
            "",
            f"Measurement: warmup={args.warmup}, iterations={args.iters}, "
            f"cold_l2={not args.warm_l2}, mixed summaries, BF16 k_bar.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This benchmark requires one exclusive CUDA GPU")
    if args.samples < 1 or args.iters < 1 or args.warmup < 0:
        raise ValueError("samples/iters must be positive and warmup non-negative")
    if any(block not in (16, 32, 64) for block in args.block_sizes):
        raise ValueError("block sizes must be 16, 32, or 64")
    set_seed(args.seed)
    _set_runtime_environment()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(_model_args(args))
    repo, dataset = load_longbench_v2(args.hf_repo, args.split)
    num_layers = int(model.config.num_hidden_layers)
    capture_layers = list(range(num_layers)) if args.layers is None else args.layers
    invalid_layers = sorted(set(capture_layers) - set(range(num_layers)))
    if invalid_layers:
        raise ValueError(f"Invalid layers: {invalid_layers}")
    max_position = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    prepared = _prepare_inputs(dataset, tokenizer, args, max_position)
    l2_flush = None
    if not args.warm_l2:
        l2_flush = torch.empty(
            args.l2_flush_mib * 1024 * 1024,
            device=args.device,
            dtype=torch.uint8,
        )

    rows = []
    raw_path = args.output_dir / "layers.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for block_size in sorted(set(args.block_sizes)):
            for eps in sorted(set(args.eps)):
                for item in prepared:
                    input_ids = item["input_ids"].to(args.device)
                    attention_mask = item["attention_mask"].to(args.device)
                    captures = capture_first_decode_step(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        block_size=block_size,
                        eps=eps,
                        capture_layers=capture_layers,
                    )
                    sample_key = (
                        f"{item['target_context_tokens']}:{item['sample_idx']}:"
                        f"{item['record_id']}"
                    )
                    for layer_idx in capture_layers:
                        capture = captures[layer_idx]
                        measured = _benchmark_capture(
                            capture,
                            block_size=block_size,
                            eps=eps,
                            args=args,
                            l2_flush=l2_flush,
                        )
                        row = {
                            "hf_repo": repo,
                            "target_context_tokens": item["target_context_tokens"],
                            "input_tokens": int(input_ids.shape[1]),
                            "sample_idx": item["sample_idx"],
                            "sample_key": sample_key,
                            "record_index": item["record_index"],
                            "record_id": item["record_id"],
                            "record_length": item["record_length"],
                            "layer_idx": layer_idx,
                            "block_size": block_size,
                            "eps": eps,
                            "suffix_tokens": capture.suffix_tokens,
                            **measured,
                        }
                        rows.append(row)
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                    print(
                        json.dumps(
                            {
                                "context": item["target_context_tokens"],
                                "sample": item["sample_idx"],
                                "block": block_size,
                                "eps": eps,
                                "layers_complete": len(capture_layers),
                            }
                        ),
                        flush=True,
                    )
                    del captures, input_ids, attention_mask
                    torch.cuda.empty_cache()

    summaries = _summarize(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "model": args.model,
                    "repo": repo,
                    "contexts": args.contexts,
                    "samples": args.samples,
                    "block_sizes": args.block_sizes,
                    "eps": args.eps,
                    "layers": capture_layers,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "cold_l2": not args.warm_l2,
                    "mixed_summaries": True,
                    "bf16_k_bar": True,
                },
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "RESULTS.md", summaries, args, repo)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
