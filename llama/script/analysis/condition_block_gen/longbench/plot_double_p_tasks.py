import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import DEFAULT_RESULT_ROOT
from .run_all import LONGBENCH_EN_CODE_DATASETS


CURVE_METHODS = {
    "kvpress_streamllm": {
        "label": "StreamLLM",
        "color": "#4C78A8",
        "marker": "o",
    },
    "kvpress_snapkv": {
        "label": "SnapKV",
        "color": "#F58518",
        "marker": "o",
    },
    "kvpress_adakv_snapkv": {
        "label": "AdaKV SnapKV",
        "color": "#B279A2",
        "marker": "s",
    },
    "quest": {
        "label": "QUEST",
        "color": "#59A14F",
        "marker": "o",
    },
}
CONDITION_METHODS = {"condition_block", "condition_block_triton"}
CONDITION_STYLE = {
    "label": "ConditionBlock Pareto",
    "color": "#E45756",
    "marker": "o",
}
DOUBLE_P_COLOR = "#0891B2"
FULL_COLOR = "#111827"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one clean comparison panel per LongBench task from the cached "
            "eval summaries."
        )
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--datasets", nargs="+", default=LONGBENCH_EN_CODE_DATASETS)
    parser.add_argument("--p1", type=float, default=0.95)
    parser.add_argument("--p2", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def budget(row):
    value = row.get("effective_decode_budget")
    if not finite_number(value):
        value = row.get("budget")
    return float(value) if finite_number(value) and float(value) > 0 else None


def valid_point(row):
    return budget(row) is not None and finite_number(row.get("score"))


def best_at_each_budget(rows):
    by_budget = {}
    for row in rows:
        if not valid_point(row):
            continue
        point_budget = budget(row)
        previous = by_budget.get(point_budget)
        if previous is None or float(row["score"]) > float(previous["score"]):
            by_budget[point_budget] = row
    return [by_budget[key] for key in sorted(by_budget)]


def pareto_frontier(rows):
    frontier = []
    best_score = float("-inf")
    for row in best_at_each_budget(rows):
        score = float(row["score"])
        if score > best_score:
            frontier.append(row)
            best_score = score
    return frontier


def select_double_p(rows, p1, p2):
    matches = [
        row
        for row in rows
        if row.get("method") == "double_p"
        and math.isclose(float(row.get("p1", -1)), p1)
        and math.isclose(float(row.get("p2", -1)), p2)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Double-P p1={p1:g}, p2={p2:g} row, found {len(matches)}"
        )
    return matches[0]


def load_dataset(root, dataset, p1, p2):
    path = root / dataset / "eval_plots" / "eval_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run longbench.eval_dataset_plot for {dataset} first"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    full_rows = [row for row in rows if row.get("method") == "full"]
    if len(full_rows) != 1:
        raise ValueError(f"Expected one full-attention row for {dataset}")
    return {
        "dataset": dataset,
        "rows": rows,
        "full": full_rows[0],
        "double_p": select_double_p(rows, p1, p2),
    }


def plot_curve(ax, rows, style):
    points = best_at_each_budget(rows)
    if not points:
        return
    ax.plot(
        [budget(row) for row in points],
        [float(row["score"]) for row in points],
        color=style["color"],
        marker=style["marker"],
        markersize=4.0,
        linewidth=1.35,
        label=style["label"],
        zorder=2,
    )


def draw_panel(ax, task, *, annotate=True, title_size=10):
    rows = task["rows"]
    full = task["full"]
    double_p = task["double_p"]
    for method, style in CURVE_METHODS.items():
        plot_curve(ax, [row for row in rows if row.get("method") == method], style)

    condition_points = pareto_frontier(
        [row for row in rows if row.get("method") in CONDITION_METHODS]
    )
    if condition_points:
        ax.plot(
            [budget(row) for row in condition_points],
            [float(row["score"]) for row in condition_points],
            color=CONDITION_STYLE["color"],
            marker=CONDITION_STYLE["marker"],
            markersize=4.2,
            linewidth=1.6,
            label=CONDITION_STYLE["label"],
            zorder=3,
        )

    ax.axhline(
        float(full["score"]),
        color=FULL_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Full attention",
        zorder=1,
    )
    dp_budget = budget(double_p)
    dp_score = float(double_p["score"])
    ax.scatter(
        [dp_budget],
        [dp_score],
        marker="D",
        s=48,
        color=DOUBLE_P_COLOR,
        edgecolor="white",
        linewidth=0.5,
        label="Double-P",
        zorder=5,
    )
    if annotate:
        ax.annotate(
            f"DP {dp_score:.2f} @ {dp_budget:.3f}",
            (dp_budget, dp_score),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color=DOUBLE_P_COLOR,
        )
    ax.set_title(
        f"{task['dataset']}\nDP {dp_score:.2f} | Dense {float(full['score']):.2f}",
        fontsize=title_size,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.025, 1.08)
    ax.grid(True, which="both", alpha=0.22)


def ordered_legend(tasks):
    present = {row.get("method") for task in tasks for row in task["rows"]}
    labels = []
    for method, style in CURVE_METHODS.items():
        if method in present:
            labels.append(style["label"])
    if present & CONDITION_METHODS:
        labels.append(CONDITION_STYLE["label"])
    labels.extend(["Double-P", "Full attention"])
    return labels


def legend_handles(axes, labels):
    if hasattr(axes, "get_legend_handles_labels"):
        axes = [axes]
    by_label = {}
    for ax in axes:
        handles, current_labels = ax.get_legend_handles_labels()
        for label, handle in zip(current_labels, handles):
            by_label.setdefault(label, handle)
    return [by_label[label] for label in labels if label in by_label]


def plot_grid(path, tasks, dpi):
    columns = 4
    rows = math.ceil(len(tasks) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 3.75 * rows),
        constrained_layout=False,
        squeeze=False,
    )
    for index, task in enumerate(tasks):
        row, column = divmod(index, columns)
        ax = axes[row][column]
        draw_panel(ax, task)
        if column == 0:
            ax.set_ylabel("LongBench score")
        if row == rows - 1:
            ax.set_xlabel("Effective decode attention budget")
    for index in range(len(tasks), rows * columns):
        row, column = divmod(index, columns)
        axes[row][column].set_visible(False)

    labels = ordered_legend(tasks)
    handles = legend_handles(axes.flat, labels)
    visible_labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles,
        visible_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=len(handles),
        fontsize=10,
        frameon=False,
    )
    fig.suptitle(
        "LongBench per-task comparison (log decode budget)",
        fontsize=17,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.055,
        top=0.91,
        hspace=0.38,
        wspace=0.20,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_individual(path, task, dpi):
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    draw_panel(ax, task, title_size=13)
    ax.set_xlabel("Effective decode attention budget")
    ax.set_ylabel("LongBench score")
    labels = ordered_legend([task])
    handles = legend_handles(ax, labels)
    visible_labels = [handle.get_label() for handle in handles]
    ax.legend(handles, visible_labels, fontsize=8, loc="best")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.result_root / args.model / args.benchmark
    output_dir = args.output_dir or root / "double_p_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"p1={args.p1:g}_p2={args.p2:g}"
    tasks = [load_dataset(root, dataset, args.p1, args.p2) for dataset in args.datasets]

    grid_path = output_dir / f"{stem}_per_task_comparison.png"
    individual_dir = output_dir / f"{stem}_per_task"
    individual_dir.mkdir(parents=True, exist_ok=True)
    plot_grid(grid_path, tasks, args.plot_dpi)
    for task in tasks:
        plot_individual(
            individual_dir / f"{task['dataset']}.png",
            task,
            args.plot_dpi,
        )
    print(f"Saved per-task grid: {grid_path}")
    print(f"Saved {len(tasks)} individual plots: {individual_dir}")


if __name__ == "__main__":
    main()
