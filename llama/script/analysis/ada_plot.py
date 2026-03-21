import argparse
import math
import os

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot student PPL vs budget for the same strategy: adaptive vs fixed."
    )
    parser.add_argument("--dataset", type=str, default="wikitext_4096")
    parser.add_argument("--strategy", type=str, default="attention_topk")
    parser.add_argument("--loss-type", type=str, default="v_l2")
    parser.add_argument(
        "--adaptive-path",
        type=str,
        default=None,
        help=(
            "Path to adaptive metrics file. "
            "Default: ../result/{dataset}/adaptive/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
        ),
    )
    parser.add_argument(
        "--fixed-path",
        type=str,
        default=None,
        help=(
            "Path to fixed metrics file. "
            "Default: ../result/{dataset}/fixed/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
        ),
    )
    parser.add_argument(
        "--adaptive-baseline-path",
        type=str,
        default=None,
        help=(
            "Path to adaptive baseline file. "
            "Default: ../result/{dataset}/adaptive/{strategy}/qk_routing.pt"
        ),
    )
    parser.add_argument(
        "--fixed-baseline-path",
        type=str,
        default=None,
        help=(
            "Path to fixed baseline file. "
            "Default: ../result/{dataset}/fixed/{strategy}/qk_routing.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output image path. "
            "Default: ../result/{dataset}/compare/{strategy}/{loss_type}/student_ppl_adaptive_fixed_optimal_baseline.png"
        ),
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="Budgets to include.",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=None,
        help="Optional token count to convert std to stderr.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def default_paths(dataset: str, strategy: str, loss_type: str):
    adaptive_optimal = f"../result/{dataset}/adaptive/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
    fixed_optimal = f"../result/{dataset}/fixed/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
    adaptive_baseline = f"../result/{dataset}/adaptive/{strategy}/qk_routing.pt"
    fixed_baseline = f"../result/{dataset}/fixed/{strategy}/qk_routing.pt"
    output = (
        f"../result/{dataset}/compare/{strategy}/{loss_type}/"
        "student_ppl_adaptive_fixed_optimal_baseline.png"
    )
    return adaptive_optimal, fixed_optimal, adaptive_baseline, fixed_baseline, output


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
        if not isinstance(metric, dict):
            raise TypeError(f"Metric entry for budget {budget} is not a dict: {type(metric)}")
        normalized[float(budget)] = metric
    return normalized


def find_matching_budget_key(metrics_by_budget: dict, requested_budget: float):
    if requested_budget in metrics_by_budget:
        return requested_budget
    for budget in metrics_by_budget.keys():
        if math.isclose(float(budget), requested_budget, rel_tol=1e-9, abs_tol=1e-12):
            return budget
    return None


def student_ppl_and_err(metric: dict, token_count: int | None):
    if "student_nll" not in metric:
        raise KeyError("Missing 'student_nll' in metrics entry")

    nll = float(metric["student_nll"])
    ppl = math.exp(nll)

    stderr_key = "student_nll_stderr"
    std_key = "student_nll_std"
    if stderr_key in metric:
        nll_err = float(metric[stderr_key])
    elif std_key in metric:
        std = float(metric[std_key])
        if token_count is not None and token_count > 0:
            nll_err = std / math.sqrt(token_count)
        else:
            nll_err = std
    else:
        nll_err = 0.0

    # Delta method for exp transform
    return ppl, ppl * nll_err


def to_series(metrics_by_budget: dict, budgets: list[float], token_count: int | None, name: str):
    xs = []
    ys = []
    es = []

    missing = []
    for budget in budgets:
        if budget <= 0:
            continue

        key = find_matching_budget_key(metrics_by_budget, float(budget))
        if key is None:
            missing.append(float(budget))
            continue

        ppl, err = student_ppl_and_err(metrics_by_budget[key], token_count)
        xs.append(float(key))
        ys.append(ppl)
        es.append(err)

    if len(missing) > 0:
        print(f"[WARN] {name}: missing budgets: {', '.join(f'{b:g}' for b in missing)}")
    return xs, ys, es


def print_values(name: str, xs, ys):
    print(f"\n{name}")
    for b, v in zip(xs, ys):
        print(f"  budget={b:g}, student_ppl={v:.8f}")


def main():
    args = parse_args()

    (
        default_adaptive_optimal,
        default_fixed_optimal,
        default_adaptive_baseline,
        default_fixed_baseline,
        default_output,
    ) = default_paths(
        args.dataset,
        args.strategy,
        args.loss_type,
    )

    adaptive_path = args.adaptive_path if args.adaptive_path else default_adaptive_optimal
    fixed_path = args.fixed_path if args.fixed_path else default_fixed_optimal
    adaptive_baseline_path = (
        args.adaptive_baseline_path if args.adaptive_baseline_path else default_adaptive_baseline
    )
    fixed_baseline_path = (
        args.fixed_baseline_path if args.fixed_baseline_path else default_fixed_baseline
    )
    output_path = args.output if args.output else default_output

    adaptive_optimal_metrics = normalize_budget_metrics(load_torch_file(adaptive_path))
    fixed_optimal_metrics = normalize_budget_metrics(load_torch_file(fixed_path))
    adaptive_baseline_metrics = normalize_budget_metrics(load_torch_file(adaptive_baseline_path))
    fixed_baseline_metrics = normalize_budget_metrics(load_torch_file(fixed_baseline_path))

    x_adp_opt, y_adp_opt, e_adp_opt = to_series(
        adaptive_optimal_metrics,
        args.budgets,
        args.token_count,
        "adaptive-optimal",
    )
    x_fix_opt, y_fix_opt, e_fix_opt = to_series(
        fixed_optimal_metrics,
        args.budgets,
        args.token_count,
        "fixed-optimal",
    )
    x_adp_base, y_adp_base, e_adp_base = to_series(
        adaptive_baseline_metrics,
        args.budgets,
        args.token_count,
        "adaptive-baseline",
    )
    x_fix_base, y_fix_base, e_fix_base = to_series(
        fixed_baseline_metrics,
        args.budgets,
        args.token_count,
        "fixed-baseline",
    )

    print_values("[Adaptive / Optimal]", x_adp_opt, y_adp_opt)
    print_values("[Fixed / Optimal]", x_fix_opt, y_fix_opt)
    print_values("[Adaptive / Baseline]", x_adp_base, y_adp_base)
    print_values("[Fixed / Baseline]", x_fix_base, y_fix_base)

    if (
        len(x_adp_opt) == 0
        and len(x_fix_opt) == 0
        and len(x_adp_base) == 0
        and len(x_fix_base) == 0
    ):
        raise ValueError("No valid budget points found in adaptive/fixed optimal/baseline metrics.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if len(x_adp_opt) > 0:
        ax.errorbar(
            x_adp_opt,
            y_adp_opt,
            yerr=e_adp_opt,
            fmt="--o",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            color="C0",
            label="Adaptive Optimal",
        )

    if len(x_fix_opt) > 0:
        ax.errorbar(
            x_fix_opt,
            y_fix_opt,
            yerr=e_fix_opt,
            fmt="--o",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            color="C1",
            label="Fixed Optimal",
        )

    if len(x_adp_base) > 0:
        ax.errorbar(
            x_adp_base,
            y_adp_base,
            yerr=e_adp_base,
            fmt="-s",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            color="C0",
            label="Adaptive Baseline",
        )

    if len(x_fix_base) > 0:
        ax.errorbar(
            x_fix_base,
            y_fix_base,
            yerr=e_fix_base,
            fmt="-s",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            color="C1",
            label="Fixed Baseline",
        )

    x_values = sorted(set([*x_adp_opt, *x_fix_opt, *x_adp_base, *x_fix_base]))
    ax.set_xscale("log")
    if len(x_values) > 0:
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{x * 100:g}%" for x in x_values])
    ax.set_xlabel("Budget (log scale, %)")

    ax.set_ylabel("Student PPL")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(
            f"Student PPL vs Budget ({args.dataset}, {args.strategy}, adaptive/fixed, optimal/baseline)"
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
