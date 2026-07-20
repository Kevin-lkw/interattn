"""Summarize accuracy, completeness and observed latency for a full sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from ..eval import read_jsonl, score_rows
from ..run_all import LONGBENCH_EN_CODE_DATASETS


DEFAULT_TASK_INFO = Path(__file__).resolve().parents[1] / "task_info.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--blocks", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--datasets", nargs="+", default=LONGBENCH_EN_CODE_DATASETS)
    parser.add_argument("--task-info", type=Path, default=DEFAULT_TASK_INFO)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _mean(values):
    return sum(values) / len(values) if values else None


def _result_path(args, dataset, block_size, eps):
    return (
        args.result_root
        / args.model_name
        / "longbench"
        / dataset
        / f"condition_block_triton_block={block_size}_eps={eps:g}.jsonl"
    )


def _dataset_row(args, dataset, expected, block_size, eps):
    path = _result_path(args, dataset, block_size, eps)
    rows = read_jsonl(path) if path.exists() else []
    ids = {str(row.get("id")) for row in rows}
    complete = len(ids) >= expected
    latencies = [
        float(row["generation_seconds"])
        for row in rows
        if row.get("generation_seconds") is not None
    ]
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    total_seconds = sum(latencies)
    score = score_rows(dataset, rows, metadata={}, use_longbench_e=False) if complete else None
    return {
        "dataset": dataset,
        "block_size": block_size,
        "eps": eps,
        "expected": expected,
        "samples": len(ids),
        "complete": complete,
        "score": score,
        "mean_latency_s": _mean(latencies),
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_input_tokens": _mean([float(row.get("input_tokens", 0)) for row in rows]),
        "mean_output_tokens": _mean([float(row.get("output_tokens", 0)) for row in rows]),
        "output_tokens_per_second": output_tokens / total_seconds if total_seconds else None,
        "path": str(path),
    }


def _config_rows(dataset_rows, blocks, eps_values, dataset_count):
    summaries = []
    for block_size in blocks:
        for eps in eps_values:
            rows = [
                row
                for row in dataset_rows
                if row["block_size"] == block_size and row["eps"] == eps
            ]
            complete_rows = [row for row in rows if row["complete"]]
            all_complete = len(complete_rows) == dataset_count
            total_samples = sum(row["samples"] for row in rows)
            total_expected = sum(row["expected"] for row in rows)
            weighted_latency = sum(
                (row["mean_latency_s"] or 0.0) * row["samples"] for row in rows
            )
            total_output_tokens = sum(
                (row["mean_output_tokens"] or 0.0) * row["samples"] for row in rows
            )
            summaries.append(
                {
                    "block_size": block_size,
                    "eps": eps,
                    "datasets_complete": len(complete_rows),
                    "datasets_total": dataset_count,
                    "samples": total_samples,
                    "expected_samples": total_expected,
                    "complete": all_complete,
                    "macro_score": (
                        _mean([row["score"] for row in complete_rows])
                        if all_complete
                        else None
                    ),
                    "sample_weighted_mean_latency_s": (
                        weighted_latency / total_samples if total_samples else None
                    ),
                    "mean_dataset_output_tokens_per_second": _mean(
                        [
                            row["output_tokens_per_second"]
                            for row in rows
                            if row["output_tokens_per_second"] is not None
                        ]
                    ),
                    "overall_output_tokens_per_second": (
                        total_output_tokens / weighted_latency
                        if weighted_latency
                        else None
                    ),
                }
            )
    return summaries


def _block_comparisons(config_rows, eps_values):
    indexed = {(row["block_size"], row["eps"]): row for row in config_rows}
    comparisons = []
    for eps in eps_values:
        block32 = indexed.get((32, eps))
        block64 = indexed.get((64, eps))
        if block32 is None or block64 is None:
            continue
        latency32 = block32["sample_weighted_mean_latency_s"]
        latency64 = block64["sample_weighted_mean_latency_s"]
        comparisons.append(
            {
                "eps": eps,
                "complete": block32["complete"] and block64["complete"],
                "block64_vs_block32_speedup": (
                    latency32 / latency64 if latency32 and latency64 else None
                ),
                "block64_minus_block32_macro_score": (
                    block64["macro_score"] - block32["macro_score"]
                    if block32["macro_score"] is not None
                    and block64["macro_score"] is not None
                    else None
                ),
            }
        )
    return comparisons


def _format(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _write_markdown(path, config_rows, comparisons, dataset_rows):
    lines = [
        "# Efficient condition-block full LongBench sweep",
        "",
        "Latency is observed end-to-end generation latency per sample, including prefill and decode. "
        "Stats are disabled, so routing budget is intentionally unavailable.",
        "",
        "## Configuration summary",
        "",
        "| block | eps | complete datasets | samples | macro score | mean latency/sample | overall output tok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['block_size']} | {row['eps']:g} | "
            f"{row['datasets_complete']}/{row['datasets_total']} | "
            f"{row['samples']}/{row['expected_samples']} | "
            f"{_format(row['macro_score'], 2)} | "
            f"{_format(row['sample_weighted_mean_latency_s'])} s | "
            f"{_format(row['overall_output_tokens_per_second'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Block-size comparison",
            "",
            "A speedup above 1 means block 64 is faster than block 32 at the same epsilon.",
            "",
            "| eps | complete | block64 vs block32 speedup | block64 - block32 macro score |",
            "|---:|:---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['eps']:g} | {'yes' if row['complete'] else 'no'} | "
            f"{_format(row['block64_vs_block32_speedup'])}x | "
            f"{_format(row['block64_minus_block32_macro_score'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Per-dataset results",
            "",
            "| dataset | block | eps | samples | score | mean latency | median latency | p95 latency | output tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['block_size']} | {row['eps']:g} | "
            f"{row['samples']}/{row['expected']} | {_format(row['score'], 2)} | "
            f"{_format(row['mean_latency_s'])} s | "
            f"{_format(row['median_latency_s'])} s | "
            f"{_format(row['p95_latency_s'])} s | "
            f"{_format(row['output_tokens_per_second'], 2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    task_info = json.loads(args.task_info.read_text(encoding="utf-8"))
    args.result_root = args.result_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.result_root / "sweep_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = sorted(set(args.blocks))
    eps_values = sorted(set(args.eps))
    rows = [
        _dataset_row(
            args,
            dataset,
            int(task_info[dataset]["num_test"]),
            block_size,
            eps,
        )
        for dataset in args.datasets
        for block_size in blocks
        for eps in eps_values
    ]
    configs = _config_rows(rows, blocks, eps_values, len(args.datasets))
    comparisons = _block_comparisons(configs, eps_values)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"configs": configs, "block_comparisons": comparisons, "datasets": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "datasets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(output_dir / "RESULTS.md", configs, comparisons, rows)
    print(json.dumps(configs, indent=2))


if __name__ == "__main__":
    main()
