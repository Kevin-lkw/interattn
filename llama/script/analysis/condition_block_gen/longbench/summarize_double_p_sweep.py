import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import DEFAULT_RESULT_ROOT, read_jsonl
from .eval_dataset_plot import infer_double_p_budgets, mean
from .run_all import LONGBENCH_EN_CODE_DATASETS
from .summarize_double_p import PAPER_LONGBENCH_DATASETS


DEFAULT_CONFIGS = [
    (0.85, 0.50),
    (0.90, 0.50),
    (0.85, 0.60),
    (0.90, 0.60),
    (0.95, 0.60),
    (0.85, 0.70),
    (0.95, 0.70),
]
FIXED_BASELINES = {
    "kvpress_streamllm": ("StreamLLM", "#4C78A8"),
    "kvpress_snapkv": ("SnapKV", "#F58518"),
}
DOUBLE_P_COLOR = "#0891B2"
FULL_COLOR = "#111827"
DOUBLE_P_LABEL_OFFSETS = {
    (0.85, 0.50): (-52, -18),
    (0.90, 0.50): (-50, 10),
    (0.85, 0.60): (-18, -28),
    (0.90, 0.60): (-20, 17),
    (0.95, 0.60): (8, -20),
    (0.85, 0.70): (10, 14),
    (0.95, 0.70): (10, -16),
}


def parse_config(value):
    try:
        p1, p2 = (float(part) for part in value.split(",", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("config must be formatted as p1,p2") from exc
    if not (0 < p2 <= p1 <= 1):
        raise argparse.ArgumentTypeError("config must satisfy 0 < p2 <= p1 <= 1")
    return p1, p2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the paper-aligned low-budget Double-P threshold sweep."
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--datasets", nargs="+", default=LONGBENCH_EN_CODE_DATASETS)
    parser.add_argument(
        "--configs",
        nargs="+",
        type=parse_config,
        default=DEFAULT_CONFIGS,
        metavar="P1,P2",
    )
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def run_stem(args, p1, p2):
    return (
        f"double_p_cluster={args.cluster_size}_iters={args.kmeans_iters}"
        f"_p1={p1:g}_p2={p2:g}_sink={args.sink_tokens}"
        f"_window={args.window_size}"
    )


def load_eval_rows(root, dataset):
    path = root / dataset / "eval_plots" / "eval_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run longbench.eval_dataset_plot for {dataset} first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def select_run(rows, *, method, p1=None, p2=None, budget=None):
    matches = []
    for row in rows:
        if row.get("method") != method:
            continue
        if p1 is not None and not math.isclose(float(row.get("p1", -1)), p1):
            continue
        if p2 is not None and not math.isclose(float(row.get("p2", -1)), p2):
            continue
        if budget is not None and not math.isclose(float(row.get("budget", -1)), budget):
            continue
        matches.append(row)
    if len(matches) != 1:
        details = f"method={method}, p1={p1}, p2={p2}, budget={budget}"
        raise ValueError(f"Expected one eval row for {details}; found {len(matches)}")
    return matches[0]


def summarize_config(args, root, eval_by_dataset, p1, p2, datasets):
    task_rows = []
    predictions = []
    for dataset in datasets:
        eval_rows = eval_by_dataset[dataset]
        double_p = select_run(eval_rows, method="double_p", p1=p1, p2=p2)
        full = select_run(eval_rows, method="full")
        pred_path = root / dataset / f"{run_stem(args, p1, p2)}.jsonl"
        dataset_predictions = read_jsonl(pred_path)
        if len(dataset_predictions) != int(double_p["num_predictions"]):
            raise ValueError(
                f"Prediction/eval count mismatch for {dataset}, ({p1:g},{p2:g})"
            )
        predictions.extend(dataset_predictions)
        task_rows.append(
            {
                "dataset": dataset,
                "p1": p1,
                "p2": p2,
                "score": float(double_p["score"]),
                "full_score": float(full["score"]),
                "score_delta": float(double_p["score"]) - float(full["score"]),
                "effective_budget": double_p.get("effective_budget"),
                "effective_decode_budget": double_p.get("effective_decode_budget"),
                "num_predictions": int(double_p["num_predictions"]),
            }
        )
    measured = infer_double_p_budgets(predictions)
    aggregate = {
        "p1": p1,
        "p2": p2,
        "num_datasets": len(task_rows),
        "num_predictions": len(predictions),
        "macro_score": mean(row["score"] for row in task_rows),
        "full_macro_score": mean(row["full_score"] for row in task_rows),
        "score_delta": mean(row["score_delta"] for row in task_rows),
        **measured,
    }
    return task_rows, aggregate


def fixed_baseline_macro(eval_by_dataset, datasets):
    datasets = list(datasets)
    points = []
    full_scores = [
        float(select_run(eval_by_dataset[dataset], method="full")["score"])
        for dataset in datasets
    ]
    for method, (label, _color) in FIXED_BASELINES.items():
        budgets = sorted(
            {
                float(row["budget"])
                for dataset in datasets
                for row in eval_by_dataset[dataset]
                if row.get("method") == method and row.get("budget") is not None
            }
        )
        for point_budget in budgets:
            matches = []
            for dataset in datasets:
                try:
                    matches.append(
                        select_run(
                            eval_by_dataset[dataset],
                            method=method,
                            budget=point_budget,
                        )
                    )
                except ValueError:
                    matches = []
                    break
            if matches:
                points.append(
                    {
                        "method": method,
                        "label": label,
                        "budget": point_budget,
                        "macro_score": mean(float(row["score"]) for row in matches),
                    }
                )
    return points, mean(full_scores)


def pareto_frontier(rows):
    frontier = []
    best_score = float("-inf")
    for row in sorted(rows, key=lambda item: item["effective_decode_budget"]):
        if row["macro_score"] > best_score:
            frontier.append(row)
            best_score = row["macro_score"]
    return frontier


def plot_sweep(path, aggregates, baselines, full_score, title, dpi):
    fig, ax = plt.subplots(figsize=(7.8, 5.1), constrained_layout=True)
    for method, (label, color) in FIXED_BASELINES.items():
        points = sorted(
            [row for row in baselines if row["method"] == method],
            key=lambda row: row["budget"],
        )
        if points:
            ax.plot(
                [row["budget"] for row in points],
                [row["macro_score"] for row in points],
                marker="o",
                linewidth=1.6,
                color=color,
                label=label,
            )
    frontier = pareto_frontier(aggregates)
    if len(frontier) > 1:
        ax.plot(
            [row["effective_decode_budget"] for row in frontier],
            [row["macro_score"] for row in frontier],
            linewidth=1.8,
            color=DOUBLE_P_COLOR,
            label="Double-P Pareto",
            zorder=3,
        )
    ax.scatter(
        [row["effective_decode_budget"] for row in aggregates],
        [row["macro_score"] for row in aggregates],
        marker="D",
        s=58,
        color=DOUBLE_P_COLOR,
        edgecolor="white",
        linewidth=0.6,
        label="Double-P configurations" if len(frontier) <= 1 else None,
        zorder=4,
    )
    for index, row in enumerate(
        sorted(aggregates, key=lambda item: item["effective_decode_budget"])
    ):
        offset = DOUBLE_P_LABEL_OFFSETS.get(
            (round(row["p1"], 2), round(row["p2"], 2)),
            (5, 7 if index % 2 == 0 else -12),
        )
        ax.annotate(
            f"({row['p1']:g},{row['p2']:g})",
            (row["effective_decode_budget"], row["macro_score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.2,
            color=DOUBLE_P_COLOR,
            arrowprops={
                "arrowstyle": "-",
                "color": DOUBLE_P_COLOR,
                "linewidth": 0.45,
                "alpha": 0.65,
            },
        )
    ax.axhline(
        full_score,
        color=FULL_COLOR,
        linestyle="--",
        linewidth=1.3,
        label=f"Full attention ({full_score:.2f})",
    )
    ax.set_xscale("log")
    ax.set_xlim(left=0.025, right=1.05)
    ax.set_xlabel("Effective decode attention budget")
    ax.set_ylabel("Macro-average LongBench score")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.23)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.result_root / args.model / args.benchmark
    output_dir = args.output_dir or root / "double_p_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_by_dataset = {
        dataset: load_eval_rows(root, dataset) for dataset in args.datasets
    }

    task_rows = []
    aggregates = []
    for p1, p2 in args.configs:
        config_task_rows, aggregate = summarize_config(
            args,
            root,
            eval_by_dataset,
            p1,
            p2,
            args.datasets,
        )
        task_rows.extend(config_task_rows)
        aggregates.append(aggregate)

    paper_datasets = [
        dataset for dataset in args.datasets if dataset in PAPER_LONGBENCH_DATASETS
    ]
    paper_aggregates = []
    for p1, p2 in args.configs:
        _task_rows, aggregate = summarize_config(
            args,
            root,
            eval_by_dataset,
            p1,
            p2,
            paper_datasets,
        )
        paper_aggregates.append(aggregate)

    baselines, full_score = fixed_baseline_macro(eval_by_dataset, args.datasets)
    paper_baselines, paper_full_score = fixed_baseline_macro(
        eval_by_dataset, paper_datasets
    )
    payload = {
        "config": {
            "model": args.model,
            "benchmark": args.benchmark,
            "datasets": args.datasets,
            "thresholds": [list(config) for config in args.configs],
        },
        "aggregate_16_task": aggregates,
        "aggregate_paper_13_task": paper_aggregates,
        "per_task": task_rows,
        "fixed_baselines_16_task": baselines,
        "fixed_baselines_paper_13_task": paper_baselines,
    }
    json_path = output_dir / "paper_threshold_sweep.json"
    aggregate_csv_path = output_dir / "paper_threshold_sweep.csv"
    paper_aggregate_csv_path = output_dir / "paper_threshold_sweep_13_task.csv"
    task_csv_path = output_dir / "paper_threshold_sweep_per_task.csv"
    plot_path = output_dir / "paper_threshold_sweep_16_task.png"
    paper_plot_path = output_dir / "paper_threshold_sweep_13_task.png"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(aggregate_csv_path, aggregates)
    write_csv(paper_aggregate_csv_path, paper_aggregates)
    write_csv(task_csv_path, task_rows)
    plot_sweep(
        plot_path,
        aggregates,
        baselines,
        full_score,
        f"Double-P paper-threshold sweep ({len(args.datasets)} tasks)",
        args.plot_dpi,
    )
    plot_sweep(
        paper_plot_path,
        paper_aggregates,
        paper_baselines,
        paper_full_score,
        "Double-P paper-threshold sweep (paper-aligned 13 tasks)",
        args.plot_dpi,
    )
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved aggregate CSV: {aggregate_csv_path}")
    print(f"Saved paper-subset aggregate CSV: {paper_aggregate_csv_path}")
    print(f"Saved per-task CSV: {task_csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved paper-subset plot: {paper_plot_path}")


if __name__ == "__main__":
    main()
