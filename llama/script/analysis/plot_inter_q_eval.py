import argparse
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot eval PPL vs budget from runner_inter_q_linear summary: "
            "baseline routing / qWk routing / optimal routing."
        )
    )
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=None,
        help="Eval sample start index. If omitted, uses --fit-start.",
    )
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
        help="Optional explicit path to runner_inter_q_linear_summary.pt",
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
    eval_start = args.fit_start if args.eval_start is None else args.eval_start
    sample_tag = (
        f"fit{args.fit_start}"
        if eval_start == args.fit_start
        else f"fit{args.fit_start}_eval{eval_start}"
    )
    return (
        f"../result/{args.dataset}_{sample_tag}/{adaptive_str}/{args.strategy}/"
        f"{args.loss_type}/runner_inter_q_linear/runner_inter_q_linear_summary.pt"
    )


def resolve_output_path(args, summary_path):
    if args.output_path is not None:
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return args.output_path

    out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "eval_ppl_vs_budget.png")


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


def pick_metric(entry, key):
    metric = entry.get(key, None)
    if metric is None:
        return None
    if not isinstance(metric, dict):
        return None
    if "ppl" not in metric:
        return None
    return float(metric["ppl"])


def collect_series(budget_items, key):
    budgets = []
    values = []
    for b, entry in budget_items:
        v = pick_metric(entry, key)
        if v is None:
            continue
        budgets.append(float(b))
        values.append(float(v))
    return budgets, values


def main():
    args = parse_args()
    if args.eval_start is None:
        args.eval_start = args.fit_start

    summary_path = args.summary_path or resolve_default_summary_path(args)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = torch.load(summary_path, map_location="cpu", weights_only=False)
    if not isinstance(summary, dict) or "results" not in summary:
        raise ValueError(f"Invalid summary format: {summary_path}")

    budget_items = sorted_budget_items(summary["results"])
    if len(budget_items) == 0:
        raise ValueError(f"No valid budget entries found in summary['results']: {summary_path}")

    b_base, y_base = collect_series(budget_items, "baseline")
    b_qwk, y_qwk = collect_series(budget_items, "inter_q_linear")
    b_opt, y_opt = collect_series(budget_items, "optimal")

    if len(y_qwk) == 0:
        raise ValueError("No qWk routing ppl found in summary['results']")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    if len(y_base) > 0:
        ax.plot(b_base, y_base, marker="o", linewidth=1.8, label="eval baseline routing")
    ax.plot(b_qwk, y_qwk, marker="o", linewidth=1.8, linestyle="--", label="eval qWk routing")
    if len(y_opt) > 0:
            ax.plot(b_opt, y_opt, marker="o", linewidth=1.8, label="eval optimal routing")
    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("Eval PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = resolve_output_path(args, summary_path)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Loaded summary: {summary_path}")
    print(f"Saved plot: {output_path}")
    print("eval budgets(qWk):", b_qwk)
    print("eval baseline routing:", y_base)
    print("eval qWk routing:", y_qwk)
    print("eval optimal routing:", y_opt)


if __name__ == "__main__":
    main()
