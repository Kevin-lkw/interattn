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
    parser.add_argument(
        "--strategy",
        type=str,
        default="attention_topk",
        help="Strategy to plot. Use 'all' to plot multiple strategies in one figure.",
    )
    parser.add_argument(
        "--all-strategies",
        type=str,
        nargs="+",
        default=["attention_topk", "h2o", "sink"],
        help="Strategies included when --strategy all.",
    )
    parser.add_argument("--loss-type", type=str, default="v_l2")
    parser.add_argument(
        "--metric",
        type=str,
        default="nll_gap",
        choices=["nll_gap", "sanity_kl", "student_nll", "student_ppl", "ppl_ratio"],
        help="Metric to plot on y-axis.",
    )
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
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[ 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="Budgets to include in the plot. Defaults to a fixed preset.",
    )
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


def metric_to_stderr(metric: dict, metric_key: str, token_count: int | None):
    stderr_key = f"{metric_key}_stderr"
    std_key = f"{metric_key}_std"

    if stderr_key in metric:
        return float(metric[stderr_key])
    if std_key in metric:
        std = float(metric[std_key])
        if token_count is not None and token_count > 0:
            return std / math.sqrt(token_count)
        return std
    return 0.0


def metric_value_and_stderr(metric: dict, metric_key: str, token_count: int | None):
    if metric_key in {"nll_gap", "sanity_kl", "student_nll"}:
        if metric_key not in metric:
            raise KeyError(f"Missing '{metric_key}' in metrics entry")
        value = float(metric[metric_key])
        stderr = metric_to_stderr(metric, metric_key, token_count)
        return value, stderr

    if metric_key == "student_ppl":
        if "student_nll" not in metric:
            raise KeyError("Missing 'student_nll' for student_ppl conversion")
        nll = float(metric["student_nll"])
        ppl = math.exp(nll)
        nll_stderr = metric_to_stderr(metric, "student_nll", token_count)
        # Delta method: y=exp(x), so sigma_y ~= exp(x)*sigma_x.
        ppl_stderr = ppl * nll_stderr
        return ppl, ppl_stderr

    if metric_key == "ppl_ratio":
        if "nll_gap" not in metric:
            raise KeyError("Missing 'nll_gap' for ppl_ratio conversion")
        nll_gap = float(metric["nll_gap"])
        ratio = math.exp(nll_gap)
        nll_gap_stderr = metric_to_stderr(metric, "nll_gap", token_count)
        ratio_stderr = ratio * nll_gap_stderr
        return ratio, ratio_stderr

    raise ValueError(f"Unsupported metric '{metric_key}'")


def find_matching_budget_key(metrics_by_budget: dict, requested_budget: float):
    if requested_budget in metrics_by_budget:
        return requested_budget

    for budget in metrics_by_budget.keys():
        if math.isclose(float(budget), requested_budget, rel_tol=1e-9, abs_tol=1e-12):
            return budget
    return None


def to_series(
    metrics_by_budget: dict,
    metric_key: str,
    token_count: int | None,
    name: str,
    requested_budgets: list[float] | None = None,
):
    xs = []
    ys = []
    es = []

    dropped_non_positive = 0
    missing_budgets = []

    if requested_budgets is None:
        budgets_to_iterate = sorted(metrics_by_budget.keys())
    else:
        budgets_to_iterate = requested_budgets

    for budget in budgets_to_iterate:
        if budget <= 0:
            dropped_non_positive += 1
            continue

        budget_key = budget
        if requested_budgets is not None:
            budget_key = find_matching_budget_key(metrics_by_budget, float(budget))
            if budget_key is None:
                missing_budgets.append(float(budget))
                continue

        metric = metrics_by_budget[budget_key]
        value, stderr = metric_value_and_stderr(metric, metric_key, token_count)

        xs.append(float(budget_key))
        ys.append(value)
        es.append(stderr)

    if dropped_non_positive > 0:
        print(f"[WARN] {name}: dropped {dropped_non_positive} non-positive budgets for log-scale x-axis.")
    if len(missing_budgets) > 0:
        formatted = ", ".join(f"{b:g}" for b in missing_budgets)
        print(f"[WARN] {name}: {len(missing_budgets)} requested budgets not found: {formatted}")
    return xs, ys, es


def print_metric_values(metric_name: str, x_opt, y_opt, x_base, y_base):
    print(f"\nMetric: {metric_name}")
    print("[Optimal routing]")
    for budget, value in zip(x_opt, y_opt):
        print(f"  budget={budget:g}, value={value:.8f}, abs={abs(value):.8f}")

    print("[Baseline routing]")
    for budget, value in zip(x_base, y_base):
        print(f"  budget={budget:g}, value={value:.8f}, abs={abs(value):.8f}")


def default_paths(dataset: str, strategy: str, loss_type: str, metric: str):
    optimal = f"../result/{dataset}/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
    baseline = f"../result/{dataset}/{strategy}/qk_routing.pt"
    output = f"../result/{dataset}/{strategy}/{loss_type}/layer_all/{metric}_vs_budget_compare.png"
    return optimal, baseline, output


def default_output_path_for_all(dataset: str, loss_type: str, metric: str):
    return f"../result/{dataset}/all/{loss_type}/{metric}_vs_budget_compare.png"


def strategy_list_from_args(args):
    if args.strategy == "all":
        deduped = []
        seen = set()
        for s in args.all_strategies:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        if len(deduped) == 0:
            raise ValueError("When --strategy all, --all-strategies must contain at least one strategy.")
        return deduped
    return [args.strategy]


def main():
    args = parse_args()

    strategies = strategy_list_from_args(args)

    if args.strategy == "all" and (args.optimal_path is not None or args.baseline_path is not None):
        raise ValueError("--optimal-path/--baseline-path are only supported for single strategy mode.")

    if args.strategy == "all":
        default_output = default_output_path_for_all(args.dataset, args.loss_type, args.metric)
    else:
        _, _, default_output = default_paths(
            args.dataset,
            args.strategy,
            args.loss_type,
            args.metric,
        )
    output_path = args.output if args.output else default_output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    color_map = {}
    any_points = False

    for idx, strategy in enumerate(strategies):
        default_optimal, default_baseline, _ = default_paths(
            args.dataset,
            strategy,
            args.loss_type,
            args.metric,
        )

        optimal_path = default_optimal if args.optimal_path is None else args.optimal_path
        baseline_path = default_baseline if args.baseline_path is None else args.baseline_path

        optimal_raw = load_torch_file(optimal_path)
        baseline_raw = load_torch_file(baseline_path)

        optimal_metrics = normalize_budget_metrics(optimal_raw)
        baseline_metrics = normalize_budget_metrics(baseline_raw)

        x_opt, y_opt, e_opt = to_series(
            optimal_metrics,
            args.metric,
            args.token_count,
            f"optimal/{strategy}",
            requested_budgets=args.budgets,
        )
        x_base, y_base, e_base = to_series(
            baseline_metrics,
            args.metric,
            args.token_count,
            f"baseline/{strategy}",
            requested_budgets=args.budgets,
        )

        print_metric_values(f"{args.metric} [{strategy}]", x_opt, y_opt, x_base, y_base)

        if len(x_opt) == 0 and len(x_base) == 0:
            print(f"[WARN] strategy={strategy}: no valid points, skip plotting.")
            continue

        any_points = True
        color = color_map.get(strategy)
        if color is None:
            color = f"C{idx % 10}"
            color_map[strategy] = color

        if len(x_opt) > 0:
            ax.errorbar(
                x_opt,
                y_opt,
                yerr=e_opt,
                fmt="--o",
                capsize=3,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=f"{strategy} optimal",
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
                color=color,
                label=f"{strategy} baseline",
            )

    if not any_points:
        raise ValueError("No valid positive budgets found across selected strategies.")

    ax.set_xscale("log")
    ax.set_xlabel("Budget (log scale)")
    if args.metric == "sanity_kl":
        ax.set_ylabel("KL divergence")
    elif args.metric == "student_nll":
        ax.set_ylabel("Student NLL")
    elif args.metric == "student_ppl":
        ax.set_ylabel("Student PPL")
        ax.set_yscale("log")
    elif args.metric == "ppl_ratio":
        ax.set_ylabel("PPL ratio (student / teacher)")
        ax.set_yscale("log")
    else:
        ax.set_ylabel("NLL gap (student - teacher)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    if args.title:
        ax.set_title(args.title)
    else:
        if args.metric == "sanity_kl":
            title_name = "KL divergence"
        elif args.metric == "student_nll":
            title_name = "Student NLL"
        elif args.metric == "student_ppl":
            title_name = "Student PPL"
        elif args.metric == "ppl_ratio":
            title_name = "PPL ratio (student / teacher)"
        else:
            title_name = "NLL gap"
        if args.strategy == "all":
            strategy_title = ", ".join(strategies)
        else:
            strategy_title = args.strategy
        ax.set_title(f"{title_name} vs Budget ({args.dataset}, {strategy_title})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()