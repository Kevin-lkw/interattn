import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eval import (
    DATASET2METRIC,
    DEFAULT_RESULT_ROOT,
    load_metadata,
    parse_run_name,
    read_jsonl,
    score_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one LongBench dataset and plot method comparisons."
    )
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--benchmark", default="longbench")
    parser.add_argument("--dataset", default="narrativeqa", choices=sorted(DATASET2METRIC))
    parser.add_argument("--hf-repo", default="THUDM/LongBench")
    parser.add_argument("--e", action="store_true", help="Evaluate LongBench-E predictions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <result-root>/<model>/<benchmark>/<dataset>/eval_plots.",
    )
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def infer_condition_budgets(rows):
    total_hybrid = 0
    total_available = 0
    decode_hybrid = 0
    decode_available = 0
    sample_budgets = []
    generated_steps = []
    for row in rows:
        budget = row.get("condition_block_budget")
        if not budget:
            continue
        total_hybrid += int(budget.get("hybrid_tokens", 0))
        total_available += int(budget.get("total_available", 0))
        sample_budget = row.get("condition_block_equiv_budget")
        if sample_budget is not None:
            sample_budgets.append(float(sample_budget))

        steps = len(budget.get("by_step", []))
        if steps <= 0:
            continue
        generated_steps.append(float(steps))
        row_units = float(budget.get("rows", 0)) / float(steps)
        first_step_available = int(round(row_units * float(row.get("input_tokens", 0))))
        decode_hybrid += int(budget.get("hybrid_tokens", 0)) - first_step_available
        decode_available += int(budget.get("total_available", 0)) - first_step_available

    return {
        "effective_budget": float(total_hybrid / total_available) if total_available else None,
        "effective_decode_budget": (
            float(decode_hybrid / decode_available) if decode_available > 0 else None
        ),
        "mean_sample_budget": mean(sample_budgets),
        "mean_generated_tokens": mean(generated_steps),
    }


def infer_effective_budget(run_info, rows):
    method = run_info.get("method")
    if method == "condition_block":
        return infer_condition_budgets(rows)["effective_budget"]
    if method == "full":
        return 1.0
    return run_info.get("budget")


def run_label(row):
    method = row.get("method")
    if method == "full":
        return "Full attention"
    if method == "kvpress_streamllm":
        return f"StreamLLM {row['budget']:g}"
    if method == "condition_block":
        return f"CondBlock b{row['block_size']} eps={row['eps']:g}"
    if "budget" in row:
        return f"{method} {row['budget']:g}"
    return row["run"]


def evaluate_dataset(args):
    dataset_name = f"{args.dataset}_e" if args.e else args.dataset
    dataset_dir = args.result_root / args.model / args.benchmark / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Result folder does not exist: {dataset_dir}")

    metadata = load_metadata(args.hf_repo, args.dataset, args.e)
    rows = []
    for pred_path in sorted(dataset_dir.glob("*.jsonl")):
        pred_rows = read_jsonl(pred_path)
        run_info = parse_run_name(pred_path)
        run_info["dataset"] = dataset_name
        run_info["file_mtime_utc"] = datetime.fromtimestamp(
            pred_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        run_info["file_size_bytes"] = pred_path.stat().st_size
        run_info["num_predictions"] = len(pred_rows)
        run_info["score"] = score_rows(args.dataset, pred_rows, metadata, args.e)
        if run_info.get("method") == "condition_block":
            run_info.update(infer_condition_budgets(pred_rows))
        else:
            run_info["effective_budget"] = infer_effective_budget(run_info, pred_rows)
            run_info["effective_decode_budget"] = run_info["effective_budget"]
            run_info["mean_sample_budget"] = run_info["effective_budget"]
            run_info["mean_generated_tokens"] = None
        run_info["mean_input_tokens"] = mean(
            float(row["input_tokens"]) for row in pred_rows if "input_tokens" in row
        )
        run_info["label"] = run_label(run_info)
        rows.append(run_info)
    return rows


def sort_key(row):
    method_order = {
        "full": 0,
        "kvpress_streamllm": 1,
        "h2o": 2,
        "quest": 3,
        "condition_block": 4,
    }
    return (
        method_order.get(row.get("method"), 99),
        row.get("budget", -1),
        row.get("block_size", -1),
        row.get("eps", -1),
        row["run"],
    )


def write_json(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = [
        "dataset",
        "run",
        "label",
        "method",
        "score",
        "effective_budget",
        "effective_decode_budget",
        "mean_sample_budget",
        "budget",
        "block_size",
        "eps",
        "num_predictions",
        "mean_generated_tokens",
        "file_mtime_utc",
        "file_size_bytes",
        "mean_input_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_score_vs_budget(path, rows, dataset, dpi):
    rows = [row for row in rows if row.get("score") is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {
        "full": "#111827",
        "kvpress_streamllm": "#4C78A8",
        "h2o": "#F58518",
        "quest": "#54A24B",
        "condition_block": "#E45756",
    }

    condition_rows = [row for row in rows if row.get("method") == "condition_block"]
    grouped = {}
    for row in condition_rows:
        grouped.setdefault(row.get("block_size"), []).append(row)
    for block_size, block_rows in sorted(grouped.items()):
        points = sorted(
            [row for row in block_rows if row.get("effective_decode_budget") is not None],
            key=lambda row: row["effective_decode_budget"],
        )
        if not points:
            continue
        ax.plot(
            [row["effective_decode_budget"] for row in points],
            [row["score"] for row in points],
            marker="o",
            linewidth=1.6,
            color=colors["condition_block"],
            alpha=0.95 if block_size == 16 else 0.55,
            label=f"CondBlock block={block_size}",
        )

    for method in ["full", "kvpress_streamllm", "h2o", "quest"]:
        points = [
            row
            for row in rows
            if row.get("method") == method and row.get("effective_decode_budget") is not None
        ]
        if not points:
            continue
        ax.scatter(
            [row["effective_decode_budget"] for row in points],
            [row["score"] for row in points],
            s=48,
            color=colors.get(method, "#777777"),
            label=method,
            zorder=3,
        )
        for row in points:
            ax.annotate(
                row["label"],
                (row["effective_decode_budget"], row["score"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(f"{dataset}: LongBench score vs decode budget")
    ax.set_xlabel("Effective decode attention budget")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_condition_eps(path, rows, dataset, dpi):
    condition_rows = [
        row
        for row in rows
        if row.get("method") == "condition_block" and row.get("score") is not None
    ]
    if not condition_rows:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    grouped = {}
    for row in condition_rows:
        grouped.setdefault(row.get("block_size"), []).append(row)

    for block_size, block_rows in sorted(grouped.items()):
        points = sorted(block_rows, key=lambda row: row.get("eps", 0.0))
        ax.plot(
            [row["eps"] for row in points],
            [row["score"] for row in points],
            marker="o",
            linewidth=1.6,
            label=f"block={block_size}",
        )
        for row in points:
            budget = row.get("effective_decode_budget")
            if budget is None:
                continue
            ax.annotate(
                f"{budget:.2f}",
                (row["eps"], row["score"]),
                xytext=(0, 6),
                ha="center",
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(f"{dataset}: condition_block eps sweep")
    ax.set_xlabel("eps")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    rows = sorted(evaluate_dataset(args), key=sort_key)
    dataset_name = f"{args.dataset}_e" if args.e else args.dataset
    dataset_dir = args.result_root / args.model / args.benchmark / dataset_name
    output_dir = args.output_dir or dataset_dir / "eval_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "eval_summary.json"
    csv_path = output_dir / "eval_summary.csv"
    budget_plot_path = output_dir / "score_vs_effective_decode_budget.png"
    eps_plot_path = output_dir / "condition_block_eps.png"

    write_json(json_path, rows)
    write_csv(csv_path, rows)
    plot_score_vs_budget(budget_plot_path, rows, dataset_name, args.plot_dpi)
    has_eps_plot = plot_condition_eps(eps_plot_path, rows, dataset_name, args.plot_dpi)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {budget_plot_path}")
    if has_eps_plot:
        print(f"Saved plot: {eps_plot_path}")
    print_warnings(rows)
    print("")
    print("Top rows:")
    scored_rows = [row for row in rows if row.get("score") is not None]
    for row in sorted(scored_rows, key=lambda item: item["score"], reverse=True)[:8]:
        budget = row.get("effective_budget")
        if row.get("method") == "condition_block":
            budget = row.get("effective_decode_budget")
        budget_text = "n/a" if budget is None else f"{budget:.4f}"
        print(f"{row['score']:6.2f}  budget={budget_text}  n={row['num_predictions']:4d}  {row['label']}")


def print_warnings(rows):
    if not rows:
        return
    complete_counts = [row["num_predictions"] for row in rows if row.get("num_predictions")]
    expected = max(complete_counts) if complete_counts else 0
    incomplete = [row for row in rows if row.get("num_predictions") != expected]
    if expected and incomplete:
        names = ", ".join(row["run"] for row in incomplete)
        print(f"WARNING incomplete runs, expected {expected} rows: {names}")

    mtimes = [
        datetime.fromisoformat(row["file_mtime_utc"]).timestamp()
        for row in rows
        if row.get("num_predictions") == expected and expected
    ]
    if mtimes and max(mtimes) - min(mtimes) > 12 * 3600:
        print(
            "WARNING result files span more than 12 hours. "
            "They may come from different code, prompt, or backend settings."
        )


if __name__ == "__main__":
    main()
