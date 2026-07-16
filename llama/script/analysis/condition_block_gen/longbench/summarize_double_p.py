import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import (
    DEFAULT_RESULT_ROOT,
    load_metadata,
    parse_run_name,
    read_jsonl,
    score_rows,
)
from .eval_dataset_plot import infer_double_p_budgets, mean
from .run_all import LONGBENCH_EN_CODE_DATASETS


PAPER_LONGBENCH_DATASETS = {
    "2wikimqa",
    "gov_report",
    "hotpotqa",
    "lcc",
    "multi_news",
    "multifieldqa_en",
    "musique",
    "narrativeqa",
    "passage_retrieval_en",
    "qasper",
    "qmsum",
    "repobench-p",
    "triviaqa",
}
FIXED_BASELINE_METHODS = {
    "kvpress_streamllm": "StreamLLM",
    "kvpress_snapkv": "SnapKV",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score and summarize one complete Double-P LongBench run."
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument("--datasets", nargs="+", default=LONGBENCH_EN_CODE_DATASETS)
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--p1", type=float, default=0.95)
    parser.add_argument("--p2", type=float, default=0.70)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize available rows instead of requiring complete test sets.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def run_stem(args):
    return (
        f"double_p_cluster={args.cluster_size}_iters={args.kmeans_iters}"
        f"_p1={args.p1:g}_p2={args.p2:g}_sink={args.sink_tokens}"
        f"_window={args.window_size}"
    )


def summarize(args):
    root = args.result_root / args.model / args.benchmark
    rows = []
    all_predictions = []
    predictions_by_dataset = {}
    baseline_scores = {method: [] for method in FIXED_BASELINE_METHODS}
    for dataset in args.datasets:
        metadata = load_metadata(args.hf_repo, dataset, False)
        pred_path = root / dataset / f"{run_stem(args)}.jsonl"
        full_path = root / dataset / "full_budget=1.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing Double-P predictions: {pred_path}")
        if not full_path.exists():
            raise FileNotFoundError(f"Missing full-attention predictions: {full_path}")

        predictions = read_jsonl(pred_path)
        full_predictions = read_jsonl(full_path)
        expected = len(metadata)
        if not args.allow_incomplete:
            if len(predictions) != expected:
                raise ValueError(
                    f"Incomplete Double-P run for {dataset}: {len(predictions)}/{expected}"
                )
            if len(full_predictions) != expected:
                raise ValueError(
                    f"Incomplete full-attention run for {dataset}: "
                    f"{len(full_predictions)}/{expected}"
                )

        budgets = infer_double_p_budgets(predictions)
        double_p_score = score_rows(dataset, predictions, metadata, False)
        full_score = score_rows(dataset, full_predictions, metadata, False)
        for method in FIXED_BASELINE_METHODS:
            for baseline_path in sorted((root / dataset).glob(f"{method}_budget=*.jsonl")):
                baseline_predictions = read_jsonl(baseline_path)
                if not args.allow_incomplete and len(baseline_predictions) != expected:
                    raise ValueError(
                        f"Incomplete {method} run for {dataset}: "
                        f"{len(baseline_predictions)}/{expected}"
                    )
                run_info = parse_run_name(baseline_path)
                baseline_scores[method].append(
                    {
                        "dataset": dataset,
                        "budget": float(run_info["budget"]),
                        "score": score_rows(
                            dataset,
                            baseline_predictions,
                            metadata,
                            False,
                        ),
                    }
                )
        rows.append(
            {
                "dataset": dataset,
                "num_predictions": len(predictions),
                "double_p_score": double_p_score,
                "full_score": full_score,
                "score_delta": (
                    round(double_p_score - full_score, 2)
                    if double_p_score is not None and full_score is not None
                    else None
                ),
                **budgets,
                "mean_input_tokens": mean(
                    float(row["input_tokens"])
                    for row in predictions
                    if row.get("input_tokens") is not None
                ),
                "mean_output_tokens": mean(
                    float(row["output_tokens"])
                    for row in predictions
                    if row.get("output_tokens") is not None
                ),
            }
        )
        all_predictions.extend(predictions)
        predictions_by_dataset[dataset] = predictions

    aggregate = aggregate_rows(
        f"macro_average_{len(rows)}_task",
        rows,
        all_predictions,
    )
    paper_rows = [row for row in rows if row["dataset"] in PAPER_LONGBENCH_DATASETS]
    paper_aggregate = None
    if len(paper_rows) == len(PAPER_LONGBENCH_DATASETS):
        paper_predictions = [
            prediction
            for row in paper_rows
            for prediction in predictions_by_dataset[row["dataset"]]
        ]
        paper_aggregate = aggregate_rows(
            "paper_macro_average_13_task",
            paper_rows,
            paper_predictions,
        )
    method_macro = build_method_macro(
        rows,
        aggregate,
        baseline_scores,
        required_datasets={row["dataset"] for row in rows},
    )
    paper_method_macro = None
    if paper_aggregate is not None:
        paper_method_macro = build_method_macro(
            paper_rows,
            paper_aggregate,
            baseline_scores,
            required_datasets=PAPER_LONGBENCH_DATASETS,
        )
    return rows, aggregate, paper_aggregate, method_macro, paper_method_macro


def aggregate_rows(label, rows, predictions):
    aggregate_budgets = infer_double_p_budgets(predictions)
    return {
        "dataset": label,
        "num_datasets": len(rows),
        "num_predictions": sum(row["num_predictions"] for row in rows),
        "double_p_score": mean(row["double_p_score"] for row in rows),
        "full_score": mean(row["full_score"] for row in rows),
        "score_delta": mean(row["score_delta"] for row in rows),
        **aggregate_budgets,
        "mean_input_tokens": mean(row["mean_input_tokens"] for row in rows),
        "mean_output_tokens": mean(row["mean_output_tokens"] for row in rows),
    }


def build_method_macro(rows, aggregate, baseline_scores, *, required_datasets):
    required_datasets = set(required_datasets)
    points = [
        {
            "method": "full",
            "label": "Full attention",
            "budget": 1.0,
            "macro_score": mean(row["full_score"] for row in rows),
            "num_datasets": len(rows),
        },
        {
            "method": "double_p",
            "label": "Double-P",
            "budget": aggregate["effective_decode_budget"],
            "macro_score": aggregate["double_p_score"],
            "num_datasets": len(rows),
        },
    ]
    for method, label in FIXED_BASELINE_METHODS.items():
        records = baseline_scores[method]
        for budget in sorted({record["budget"] for record in records}):
            budget_records = [
                record
                for record in records
                if record["budget"] == budget
                and record["dataset"] in required_datasets
            ]
            if {record["dataset"] for record in budget_records} != required_datasets:
                continue
            points.append(
                {
                    "method": method,
                    "label": label,
                    "budget": budget,
                    "macro_score": mean(record["score"] for record in budget_records),
                    "num_datasets": len(budget_records),
                }
            )
    return points


def write_csv(path, rows, aggregates):
    fields = [
        "dataset",
        "num_datasets",
        "num_predictions",
        "double_p_score",
        "full_score",
        "score_delta",
        "effective_budget",
        "effective_decode_budget",
        "mean_sample_budget",
        "mean_generated_tokens",
        "mean_input_tokens",
        "mean_output_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(aggregates)


def write_method_csv(path, rows):
    fields = ["method", "label", "budget", "macro_score", "num_datasets"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path, rows, aggregate, *, p1, p2, dpi):
    labels = [row["dataset"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    fig, (score_ax, budget_ax) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    score_ax.bar(
        [position - width / 2 for position in positions],
        [row["full_score"] for row in rows],
        width=width,
        color="#111827",
        label=f"Full attention (macro {aggregate['full_score']:.2f})",
    )
    score_ax.bar(
        [position + width / 2 for position in positions],
        [row["double_p_score"] for row in rows],
        width=width,
        color="#0891B2",
        label=f"Double-P (macro {aggregate['double_p_score']:.2f})",
    )
    score_ax.set_ylabel("LongBench score")
    score_ax.set_title(f"Double-P p1={p1:g}, p2={p2:g} vs full attention")
    score_ax.grid(axis="y", alpha=0.25)
    score_ax.legend(fontsize=9)

    budget_ax.bar(
        positions,
        [row["effective_decode_budget"] for row in rows],
        color="#0891B2",
    )
    budget_ax.axhline(
        aggregate["effective_decode_budget"],
        color="#DC2626",
        linestyle="--",
        linewidth=1.4,
        label=f"token-weighted aggregate {aggregate['effective_decode_budget']:.3f}",
    )
    budget_ax.set_ylabel("Decode budget")
    budget_ax.set_ylim(bottom=0)
    budget_ax.set_xticks(positions, labels, rotation=35, ha="right")
    budget_ax.grid(axis="y", alpha=0.25)
    budget_ax.legend(fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_method_macro(path, rows, *, title, dpi):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    colors = {
        "kvpress_streamllm": "#4C78A8",
        "kvpress_snapkv": "#F58518",
        "double_p": "#0891B2",
        "full": "#111827",
    }
    for method in FIXED_BASELINE_METHODS:
        points = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["budget"],
        )
        if not points:
            continue
        ax.plot(
            [row["budget"] for row in points],
            [row["macro_score"] for row in points],
            marker="o",
            linewidth=1.7,
            color=colors[method],
            label=points[0]["label"],
        )
    double_p = next(row for row in rows if row["method"] == "double_p")
    ax.scatter(
        [double_p["budget"]],
        [double_p["macro_score"]],
        marker="D",
        s=72,
        color=colors["double_p"],
        label="Double-P",
        zorder=4,
    )
    full = next(row for row in rows if row["method"] == "full")
    ax.axhline(
        full["macro_score"],
        color=colors["full"],
        linestyle="--",
        linewidth=1.4,
        label=f"Full attention ({full['macro_score']:.2f})",
    )
    ax.set_xscale("log")
    ax.set_xlim(left=0.04, right=1.05)
    ax.set_xlabel("Effective decode attention budget")
    ax.set_ylabel("Macro-average LongBench score")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    rows, aggregate, paper_aggregate, method_macro, paper_method_macro = summarize(args)
    root = args.result_root / args.model / args.benchmark
    output_dir = args.output_dir or root / "double_p_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"p1={args.p1:g}_p2={args.p2:g}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    method_csv_path = output_dir / f"{stem}_method_macro.csv"
    method_plot_path = output_dir / f"{stem}_method_macro.png"
    paper_method_plot_path = output_dir / f"{stem}_paper_13_task_method_macro.png"
    json_path.write_text(
        json.dumps(
            {
                "config": vars(args)
                | {
                    "result_root": str(args.result_root),
                    "output_dir": str(output_dir),
                },
                "datasets": rows,
                "aggregate": aggregate,
                "paper_13_task_aggregate": paper_aggregate,
                "method_macro_comparison": method_macro,
                "paper_13_task_method_macro_comparison": paper_method_macro,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    aggregates = [aggregate]
    if paper_aggregate is not None:
        aggregates.append(paper_aggregate)
    write_csv(csv_path, rows, aggregates)
    write_method_csv(method_csv_path, method_macro)
    plot(
        plot_path,
        rows,
        aggregate,
        p1=args.p1,
        p2=args.p2,
        dpi=args.plot_dpi,
    )
    plot_method_macro(
        method_plot_path,
        method_macro,
        title=f"LongBench {len(rows)}-task macro comparison",
        dpi=args.plot_dpi,
    )
    if paper_method_macro is not None:
        plot_method_macro(
            paper_method_plot_path,
            paper_method_macro,
            title="LongBench paper-aligned 13-task macro comparison",
            dpi=args.plot_dpi,
        )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if paper_aggregate is not None:
        print(json.dumps(paper_aggregate, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved method CSV: {method_csv_path}")
    print(f"Saved method plot: {method_plot_path}")
    if paper_method_macro is not None:
        print(f"Saved paper-subset method plot: {paper_method_plot_path}")


if __name__ == "__main__":
    main()
