import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import DEFAULT_RESULT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Plot condition_block eps vs effective budget.")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "musique",
        ],
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--budget-field",
        default="effective_decode_budget",
        choices=["effective_decode_budget", "effective_budget", "mean_sample_budget"],
        help="Budget column from eval_summary.csv to plot.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_points(args):
    points = []
    root = args.result_root / args.model / args.benchmark
    for dataset in args.datasets:
        summary_path = root / dataset / "eval_plots" / "eval_summary.csv"
        if not summary_path.exists():
            print(f"WARNING missing eval summary, skip: {summary_path}")
            continue
        for row in read_rows(summary_path):
            if row.get("method") != "condition_block":
                continue
            if row.get("block_size") != str(args.block_size):
                continue
            if not row.get("eps") or not row.get(args.budget_field):
                continue
            points.append(
                {
                    "dataset": dataset,
                    "eps": float(row["eps"]),
                    "budget": float(row[args.budget_field]),
                    "score": float(row["score"]) if row.get("score") else None,
                    "num_predictions": int(row["num_predictions"]),
                    "run": row["run"],
                }
            )
    return points


def write_csv(path, points):
    fields = ["dataset", "eps", "budget", "score", "num_predictions", "run"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def plot(path, points, dpi):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for dataset in sorted({point["dataset"] for point in points}):
        dataset_points = sorted(
            [point for point in points if point["dataset"] == dataset],
            key=lambda point: point["eps"],
        )
        ax.plot(
            [point["eps"] for point in dataset_points],
            [point["budget"] for point in dataset_points],
            marker="o",
            linewidth=1.5,
            label=dataset,
        )

    ax.set_xscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("Attention budget")
    ax.set_title("condition_block eps vs budget")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.result_root / args.model / args.benchmark / "eps_budget_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    points = collect_points(args)
    if not points:
        raise ValueError("No condition_block points found. Run eval_dataset_plot.py first.")

    csv_path = output_dir / f"condition_block_block={args.block_size}_{args.budget_field}_eps_budget.csv"
    plot_path = output_dir / f"condition_block_block={args.block_size}_{args.budget_field}_eps_budget.png"
    write_csv(csv_path, sorted(points, key=lambda item: (item["dataset"], item["eps"])))
    plot(plot_path, points, args.plot_dpi)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
