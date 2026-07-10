import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import DATASET2METRIC, load_metadata, read_jsonl, score_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot condition_block_hierarchy LongBench runs."
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=Path("llama/result/hierarchy"))
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "qasper", "trec"])
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def infer_budget(rows):
    total_hybrid = 0
    total_available = 0
    sample_budgets = []
    for row in rows:
        budget = row.get("condition_block_budget") or {}
        total_hybrid += int(budget.get("hybrid_tokens", 0))
        total_available += int(budget.get("total_available", 0))
        if row.get("condition_block_equiv_budget") is not None:
            sample_budgets.append(float(row["condition_block_equiv_budget"]))
    return {
        "effective_budget": (
            float(total_hybrid / total_available) if total_available > 0 else None
        ),
        "mean_sample_budget": mean(sample_budgets),
    }


def read_runs(args):
    rows = []
    metadata_cache = {}
    for dataset in args.datasets:
        if dataset not in DATASET2METRIC:
            raise ValueError(f"Unsupported LongBench dataset: {dataset}")
        dataset_dir = args.result_root / args.model / args.benchmark / dataset
        if not dataset_dir.exists():
            continue
        metadata_cache[dataset] = load_metadata(args.hf_repo, dataset, False)
        for pred_path in sorted(dataset_dir.glob("condition_block_hierarchy_*.jsonl")):
            pred_rows = read_jsonl(pred_path)
            if not pred_rows:
                continue
            first = pred_rows[0]
            score = score_rows(dataset, pred_rows, metadata_cache[dataset], False)
            budget = infer_budget(pred_rows)
            block_sizes = first.get("block_sizes") or first.get("condition_block_hierarchy", {}).get("block_sizes")
            eps = first.get("condition_eps")
            if eps is None:
                eps = first.get("condition_block_hierarchy", {}).get("eps")
            rows.append(
                {
                    "dataset": dataset,
                    "run": pred_path.stem,
                    "score": score,
                    "num_predictions": len(pred_rows),
                    "block_sizes": "-".join(str(size) for size in block_sizes or []),
                    "eps": float(eps) if eps is not None else None,
                    "effective_budget": budget["effective_budget"],
                    "mean_sample_budget": budget["mean_sample_budget"],
                    "mean_input_tokens": mean(
                        float(row["input_tokens"])
                        for row in pred_rows
                        if row.get("input_tokens") is not None
                    ),
                    "mean_output_tokens": mean(
                        float(row["output_tokens"])
                        for row in pred_rows
                        if row.get("output_tokens") is not None
                    ),
                }
            )
    return rows


def write_json(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = [
        "dataset",
        "run",
        "score",
        "num_predictions",
        "block_sizes",
        "eps",
        "effective_budget",
        "mean_sample_budget",
        "mean_input_tokens",
        "mean_output_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_score_by_dataset(path, rows, dpi):
    datasets = sorted({row["dataset"] for row in rows})
    eps_values = sorted({row["eps"] for row in rows if row.get("eps") is not None})
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    width = 0.8 / max(len(eps_values), 1)
    x_base = list(range(len(datasets)))
    for idx, eps in enumerate(eps_values):
        values = []
        budgets = []
        for dataset in datasets:
            matches = [row for row in rows if row["dataset"] == dataset and row["eps"] == eps]
            row = matches[0] if matches else {}
            values.append(row.get("score", 0.0) or 0.0)
            budgets.append(row.get("effective_budget"))
        offset = (idx - (len(eps_values) - 1) / 2.0) * width
        bars = ax.bar(
            [x + offset for x in x_base],
            values,
            width=width,
            label=f"eps={eps:g}",
        )
        for bar, budget in zip(bars, budgets):
            if budget is None:
                continue
            ax.annotate(
                f"{budget:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x_base)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("LongBench score")
    ax.set_title("condition_block_hierarchy LongBench subset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_score_vs_budget(path, rows, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for dataset in sorted({row["dataset"] for row in rows}):
        points = [
            row
            for row in rows
            if row["dataset"] == dataset and row.get("effective_budget") is not None
        ]
        points.sort(key=lambda row: row["effective_budget"])
        if not points:
            continue
        ax.plot(
            [row["effective_budget"] for row in points],
            [row["score"] for row in points],
            marker="o",
            linewidth=1.5,
            label=dataset,
        )
        for row in points:
            if row.get("eps") is not None:
                ax.annotate(
                    f"eps={row['eps']:g}",
                    (row["effective_budget"], row["score"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel("Effective attention budget")
    ax.set_ylabel("LongBench score")
    ax.set_title("Score vs effective budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.result_root / args.model / "hierarchy_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_runs(args), key=lambda row: (row["dataset"], row.get("eps") or 0.0))
    write_json(output_dir / "longbench_hierarchy_summary.json", rows)
    write_csv(output_dir / "longbench_hierarchy_summary.csv", rows)
    if rows:
        plot_score_by_dataset(output_dir / "longbench_hierarchy_scores.png", rows, args.plot_dpi)
        plot_score_vs_budget(output_dir / "longbench_hierarchy_score_vs_budget.png", rows, args.plot_dpi)
    print(f"Saved {len(rows)} rows to: {output_dir}")


if __name__ == "__main__":
    main()
