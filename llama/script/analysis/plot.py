import argparse
import math
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot NLL gap vs budget for optimal routing and baseline routing, with error bars."
        )
    )
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--strategy", type=str, default="attention_topk")
    parser.add_argument("--loss-type", type=str, default="v_l2")
    parser.add_argument(
        "--optimal-path",
        type=str,
        default=None,
        help=(
            "Path to optimal result file. Default: ../result/{dataset}/{strategy}/{loss_type}/"
            "layer_all/budget_to_final_metrics.pt"
        ),
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help=(
            "Path to baseline result file. Default: ../result/{dataset}/{strategy}/qk_routing.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output image path. Default: ../result/{dataset}/{strategy}/{loss_type}/"
            "layer_all/nll_gap_vs_budget_compare.png"
        ),
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=None,
        help=(
            "Token sample count used to convert stored nll_gap_std into standard error. "
            "If omitted, nll_gap_std is used directly as error bar."
        ),
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_torch_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_budget_metrics(raw):
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict-like metrics object, got: {type(raw)}")

    if "budgets" in raw:
        raw = raw["budgets"]

    normalized = {}
    for budget, metric in raw.items():
        b = float(budget)
        if not isinstance(metric, dict):
            raise TypeError(f"Metric entry for budget {budget} is not a dict: {type(metric)}")
        normalized[b] = metric
    return normalized


def metric_to_stderr(metric: dict, token_count: int | None):
    if "nll_gap_stderr" in metric:
        return float(metric["nll_gap_stderr"])
    if "nll_gap_std" in metric:
        std = float(metric["nll_gap_std"])
        if token_count is not None and token_count > 0:
            return std / math.sqrt(token_count)
        return std
    return 0.0


def to_series(metrics_by_budget: dict, token_count: int | None, name: str):
    xs = []
    ys = []
    es = []

    dropped_non_positive = 0
    for budget in sorted(metrics_by_budget.keys()):
        if budget <= 0:
            dropped_non_positive += 1
            continue

        metric = metrics_by_budget[budget]
        if "nll_gap" not in metric:
            raise KeyError(f"Missing 'nll_gap' in {name} metrics for budget={budget}")

        xs.append(float(budget))
        ys.append(float(metric["nll_gap"]))
        es.append(metric_to_stderr(metric, token_count))

    if dropped_non_positive > 0:
        print(f"[WARN] {name}: dropped {dropped_non_positive} non-positive budgets for log-scale x-axis.")
    return xs, ys, es


def default_paths(dataset: str, strategy: str, loss_type: str):
    optimal = f"../result/{dataset}/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
    baseline = f"../result/{dataset}/{strategy}/qk_routing.pt"
    output = f"../result/{dataset}/{strategy}/{loss_type}/layer_all/nll_gap_vs_budget_compare.png"
    return optimal, baseline, output


def main():
    args = parse_args()

    default_optimal, default_baseline, default_output = default_paths(
        args.dataset,
        args.strategy,
        args.loss_type,
    )

    optimal_path = args.optimal_path if args.optimal_path else default_optimal
    baseline_path = args.baseline_path if args.baseline_path else default_baseline
    output_path = args.output if args.output else default_output

    optimal_raw = load_torch_file(optimal_path)
    baseline_raw = load_torch_file(baseline_path)

    optimal_metrics = normalize_budget_metrics(optimal_raw)
    baseline_metrics = normalize_budget_metrics(baseline_raw)

    x_opt, y_opt, e_opt = to_series(optimal_metrics, args.token_count, "optimal")
    x_base, y_base, e_base = to_series(baseline_metrics, args.token_count, "baseline")

    if len(x_opt) == 0 and len(x_base) == 0:
        raise ValueError("No valid positive budgets found in either optimal or baseline results.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if len(x_opt) > 0:
        ax.errorbar(
            x_opt,
            y_opt,
            yerr=e_opt,
            fmt="-o",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            label="Optimal routing",
        )
    if len(x_base) > 0:
        ax.errorbar(
            x_base,
            y_base,
            yerr=e_base,
            fmt="-s",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            label="Baseline routing",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Budget (log scale)")
    ax.set_ylabel("NLL gap (student - teacher)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"NLL gap vs Budget ({args.dataset}, {args.strategy})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()