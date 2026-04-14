import argparse
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PPL vs budget from runner_inter summary (baseline/inter/optimal)."
    )
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument(
        "--replace-k-percent",
        type=float,
        default=None,
        help="Single K%% value (kept for backward compatibility).",
    )
    parser.add_argument(
        "--replace-k-percents",
        type=float,
        nargs="+",
        default=None,
        help="Multiple K%% values. One inter curve will be plotted for each K.",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional explicit path to one runner_inter_summary.pt",
    )
    parser.add_argument(
        "--summary-paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit paths for multiple runner_inter_summary.pt files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output png path",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_default_summary_path(args, replace_k_percent):
    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    return (
        f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
        f"{args.loss_type}/runner_inter/kpct_{replace_k_percent:g}/runner_inter_summary.pt"
    )


def resolve_output_path(args, summary_path):
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        return args.output_path

    out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "ppl_vs_budget.png")


def _sorted_budget_items(results_dict):
    # results_dict keys can be float or stringified float after serialization.
    items = []
    for k, v in results_dict.items():
        try:
            budget = float(k)
        except (TypeError, ValueError):
            continue
        items.append((budget, v))
    items.sort(key=lambda x: x[0])
    return items


def _resolve_k_list(args):
    if args.replace_k_percents is not None and len(args.replace_k_percents) > 0:
        return [float(x) for x in args.replace_k_percents]
    if args.replace_k_percent is not None:
        return [float(args.replace_k_percent)]
    return [50.0]


def main():
    args = parse_args()

    if args.summary_paths is not None and len(args.summary_paths) > 0:
        summary_paths = list(args.summary_paths)
        k_list = _resolve_k_list(args)
        if len(k_list) == 1 and len(summary_paths) > 1:
            # If only one K is provided but multiple summaries are given, auto-index labels.
            k_labels = [f"summary{i}" for i in range(len(summary_paths))]
        elif len(k_list) == len(summary_paths):
            k_labels = [f"K={k:g}%" for k in k_list]
        else:
            raise ValueError(
                "When using --summary-paths, provide either one K or the same number of K values."
            )
    elif args.summary_path is not None:
        summary_paths = [args.summary_path]
        k_list = _resolve_k_list(args)
        if len(k_list) != 1:
            raise ValueError("--summary-path only supports one summary; use --summary-paths for multiple")
        k_labels = [f"K={k_list[0]:g}%"]
    else:
        k_list = _resolve_k_list(args)
        summary_paths = [resolve_default_summary_path(args, k) for k in k_list]
        k_labels = [f"K={k:g}%" for k in k_list]

    loaded = []
    for path in summary_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Summary file not found: {path}")
        summary = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(summary, dict) or "results" not in summary:
            raise ValueError(f"Invalid summary file format: {path}")

        budget_items = _sorted_budget_items(summary["results"])
        if len(budget_items) == 0:
            raise ValueError(f"No valid budget entries found in summary['results']: {path}")

        budgets = [x[0] for x in budget_items]
        baseline_ppl = [float(x[1]["baseline"]["ppl"]) for x in budget_items]
        inter_ppl = [float(x[1]["inter"]["ppl"]) for x in budget_items]
        optimal_ppl = [float(x[1]["optimal"]["ppl"]) for x in budget_items]
        loaded.append(
            {
                "path": path,
                "budgets": budgets,
                "baseline": baseline_ppl,
                "inter": inter_ppl,
                "optimal": optimal_ppl,
            }
        )

    # Baseline/optimal are expected to be shared across K; we plot from the first summary.
    ref = loaded[0]
    budgets = ref["budgets"]
    baseline_ppl = ref["baseline"]
    optimal_ppl = ref["optimal"]

    for item in loaded[1:]:
        if item["budgets"] != budgets:
            raise ValueError("Budget grids are inconsistent across summaries.")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(budgets, baseline_ppl, marker="o", linewidth=1.8, label="baseline")
    ax.plot(budgets, optimal_ppl, marker="o", linewidth=1.8, label="optimal")
    for idx, item in enumerate(loaded):
        ax.plot(
            budgets,
            item["inter"],
            marker="o",
            linewidth=1.8,
            linestyle="--",
            label=f"inter ({k_labels[idx]})",
        )

    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = resolve_output_path(args, loaded[0]["path"])
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print("Loaded summaries:")
    for item in loaded:
        print("-", item["path"])
    print(f"Saved plot: {output_path}")
    print("Budgets:", budgets)
    print("baseline:", baseline_ppl)
    for idx, item in enumerate(loaded):
        print(f"inter ({k_labels[idx]}):", item["inter"])
    print("optimal:", optimal_ppl)


if __name__ == "__main__":
    main()
