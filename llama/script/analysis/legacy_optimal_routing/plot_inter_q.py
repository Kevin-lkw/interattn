import argparse
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PPL vs budget from runner_inter_q summary (baseline/inter_q/optimal)."
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
        "--summary-path",
        type=str,
        default=None,
        help="Optional explicit path to runner_inter_q_summary.pt",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output png path",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_default_summary_path(args):
    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    return (
        f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
        f"{args.loss_type}/runner_inter_q_linear/runner_inter_q_linear_summary.pt"
    )


def resolve_output_path(args, summary_path):
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return args.output_path

    out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "ppl_vs_budget.png")


def sorted_budget_items(results_dict):
    items = []
    for k, v in results_dict.items():
        try:
            budget = float(k)
        except (TypeError, ValueError):
            continue
        items.append((budget, v))
    items.sort(key=lambda x: x[0])
    return items


def main():
    args = parse_args()

    summary_path = args.summary_path or resolve_default_summary_path(args)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = torch.load(summary_path, map_location="cpu", weights_only=False)
    if not isinstance(summary, dict) or "results" not in summary:
        raise ValueError(f"Invalid summary format: {summary_path}")

    budget_items = sorted_budget_items(summary["results"])
    if len(budget_items) == 0:
        raise ValueError(f"No valid budget entries found in summary['results']: {summary_path}")

    budgets = [x[0] for x in budget_items]
    baseline_ppl = [float(x[1]["baseline"]["ppl"]) for x in budget_items]
    inter_q_ppl = [float(x[1]["inter_q_linear"]["ppl"]) for x in budget_items]
    optimal_ppl = [float(x[1]["optimal"]["ppl"]) for x in budget_items]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(budgets, baseline_ppl, marker="o", linewidth=1.8, label="baseline qk routing")
    ax.plot(budgets, inter_q_ppl, marker="o", linewidth=1.8, linestyle="--", label="q bias scaling")
    ax.plot(budgets, optimal_ppl, marker="o", linewidth=1.8, label="optimal routing")

    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = resolve_output_path(args, summary_path)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Loaded summary: {summary_path}")
    print(f"Saved plot: {output_path}")
    print("Budgets:", budgets)
    print("baseline qk routing:", baseline_ppl)
    print("q bias scaling:", inter_q_ppl)
    print("optimal routing:", optimal_ppl)


if __name__ == "__main__":
    main()
