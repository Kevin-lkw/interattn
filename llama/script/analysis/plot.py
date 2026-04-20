import argparse
import math
import os

import matplotlib.pyplot as plt
import torch


PLOT_KINDS = ["routing_gap", "inter", "inter_q", "inter_q_eval", "inter_avgkv"]
INTER_Q_VARIANTS = ["linear", "bias"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified plotting entry for routing experiments. "
            "Use --plot-kind to choose which figure type to generate."
        )
    )
    parser.add_argument(
        "--plot-kind",
        type=str,
        default="routing_gap",
        choices=PLOT_KINDS,
        help="Plot target: routing_gap | inter | inter_q | inter_q_eval | inter_avgkv",
    )

    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0, help="Used by inter/inter_q modes")
    parser.add_argument("--fit-start", type=int, default=0, help="Used by inter_q_eval mode")
    parser.add_argument("--eval-start", type=int, default=None, help="Used by inter_q_eval mode")
    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink", "all"],
    )
    parser.add_argument(
        "--all-strategies",
        type=str,
        nargs="+",
        default=["attention_topk", "h2o", "sink"],
        help="Strategies included when --strategy all (routing_gap mode).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )
    parser.add_argument("--adaptive-budget", action="store_true")

    parser.add_argument(
        "--metric",
        type=str,
        default="student_ppl",
        choices=["nll_gap", "sanity_kl", "student_nll", "student_ppl", "ppl_ratio"],
        help="Y-axis metric in routing_gap mode.",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="Budgets to include. routing_gap uses fixed preset by default.",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=None,
        help="Token sample count for converting std to stderr in routing_gap mode.",
    )

    parser.add_argument(
        "--replace-k-percent",
        type=float,
        default=None,
        help="Single K%% for inter mode (backward compatibility).",
    )
    parser.add_argument(
        "--replace-k-percents",
        type=float,
        nargs="+",
        default=None,
        help="Multiple K%% values for inter mode.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="linear",
        choices=INTER_Q_VARIANTS,
        help="Inter-q variant for inter_q_eval mode.",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Single explicit summary path.",
    )
    parser.add_argument(
        "--summary-paths",
        type=str,
        nargs="+",
        default=None,
        help="Multiple explicit summary paths (inter mode).",
    )
    parser.add_argument(
        "--optimal-path",
        type=str,
        default=None,
        help="Explicit optimal result path (routing_gap single-strategy mode).",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Explicit baseline result path (routing_gap single-strategy mode).",
    )

    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        type=str,
        default=None,
        help="Optional explicit output image path.",
    )

    return parser.parse_args()


def load_torch_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def adaptive_tag(adaptive_budget: bool):
    return "adaptive" if adaptive_budget else "fixed"


def ensure_output_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def sort_budget_items(results_dict):
    items = []
    for k, v in results_dict.items():
        try:
            budget = float(k)
        except (TypeError, ValueError):
            continue
        items.append((budget, v))
    items.sort(key=lambda x: x[0])
    return items


# ---------------------------- routing_gap mode ----------------------------
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
        return ppl, ppl * nll_stderr

    if metric_key == "ppl_ratio":
        if "nll_gap" not in metric:
            raise KeyError("Missing 'nll_gap' for ppl_ratio conversion")
        nll_gap = float(metric["nll_gap"])
        ratio = math.exp(nll_gap)
        nll_gap_stderr = metric_to_stderr(metric, "nll_gap", token_count)
        return ratio, ratio * nll_gap_stderr

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
    xs, ys, es = [], [], []
    dropped_non_positive = 0
    missing_budgets = []

    budgets_to_iterate = sorted(metrics_by_budget.keys()) if requested_budgets is None else requested_budgets

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


def default_paths_routing_gap(dataset: str, adaptive_budget: bool, strategy: str, loss_type: str, metric: str):
    ada = adaptive_tag(adaptive_budget)
    optimal = f"../result/{dataset}/{ada}/{strategy}/{loss_type}/layer_all/budget_to_final_metrics.pt"
    baseline = f"../result/{dataset}/{ada}/{strategy}/qk_routing.pt"
    output = f"../result/{dataset}/{ada}/{strategy}/{loss_type}/layer_all/{metric}_vs_budget_compare.png"
    return optimal, baseline, output


def default_output_path_all_routing_gap(dataset: str, adaptive_budget: bool, loss_type: str, metric: str):
    ada = adaptive_tag(adaptive_budget)
    return f"../result/{dataset}/{ada}/all/{loss_type}/{metric}_vs_budget_compare.png"


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


def run_routing_gap_plot(args):
    strategies = strategy_list_from_args(args)

    if args.strategy == "all" and (args.optimal_path is not None or args.baseline_path is not None):
        raise ValueError("--optimal-path/--baseline-path are only supported for single strategy mode.")

    if args.strategy == "all":
        default_output = default_output_path_all_routing_gap(
            args.dataset, args.adaptive_budget, args.loss_type, args.metric
        )
    else:
        _, _, default_output = default_paths_routing_gap(
            args.dataset,
            args.adaptive_budget,
            args.strategy,
            args.loss_type,
            args.metric,
        )
    output_path = args.output_path if args.output_path else default_output
    ensure_output_dir(output_path)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    any_points = False
    plotted_x_values = []

    for idx, strategy in enumerate(strategies):
        default_optimal, default_baseline, _ = default_paths_routing_gap(
            args.dataset,
            args.adaptive_budget,
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
        color = f"C{idx % 10}"

        if len(x_opt) > 0:
            plotted_x_values.extend(x_opt)
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
            plotted_x_values.extend(x_base)
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
    if len(plotted_x_values) > 0:
        x_ticks = sorted(set(float(x) for x in plotted_x_values))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x * 100:g}%" for x in x_ticks])
    ax.set_xlabel("Budget (log scale, %)" )
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
        strategy_title = ", ".join(strategies) if args.strategy == "all" else args.strategy
        ax.set_title(f"{title_name} vs Budget ({args.dataset}, {strategy_title})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure to: {output_path}")

    if args.show:
        plt.show()


# ---------------------------- inter mode ----------------------------
def resolve_k_list(args):
    if args.replace_k_percents is not None and len(args.replace_k_percents) > 0:
        return [float(x) for x in args.replace_k_percents]
    if args.replace_k_percent is not None:
        return [float(args.replace_k_percent)]
    return [50.0]


def default_inter_summary_path(args, replace_k_percent):
    ada = adaptive_tag(args.adaptive_budget)
    return (
        f"../result/{args.dataset}_{args.start}/{ada}/{args.strategy}/"
        f"{args.loss_type}/runner_inter/kpct_{replace_k_percent:g}/runner_inter_summary.pt"
    )


def resolve_output_for_summary(output_path, summary_path, fallback_name):
    if output_path is not None:
        ensure_output_dir(output_path)
        return output_path

    out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fallback_name)


def run_inter_plot(args):
    if args.summary_paths is not None and len(args.summary_paths) > 0:
        summary_paths = list(args.summary_paths)
        k_list = resolve_k_list(args)
        if len(k_list) == 1 and len(summary_paths) > 1:
            k_labels = [f"summary{i}" for i in range(len(summary_paths))]
        elif len(k_list) == len(summary_paths):
            k_labels = [f"K={k:g}%" for k in k_list]
        else:
            raise ValueError(
                "When using --summary-paths, provide either one K or the same number of K values."
            )
    elif args.summary_path is not None:
        summary_paths = [args.summary_path]
        k_list = resolve_k_list(args)
        if len(k_list) != 1:
            raise ValueError("--summary-path only supports one summary; use --summary-paths for multiple")
        k_labels = [f"K={k_list[0]:g}%"]
    else:
        k_list = resolve_k_list(args)
        summary_paths = [default_inter_summary_path(args, k) for k in k_list]
        k_labels = [f"K={k:g}%" for k in k_list]

    loaded = []
    for path in summary_paths:
        summary = load_torch_file(path)
        if not isinstance(summary, dict) or "results" not in summary:
            raise ValueError(f"Invalid summary file format: {path}")

        budget_items = sort_budget_items(summary["results"])
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

    output_path = resolve_output_for_summary(args.output_path, loaded[0]["path"], "ppl_vs_budget.png")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print("Loaded summaries:")
    for item in loaded:
        print("-", item["path"])
    print(f"Saved plot: {output_path}")


# ---------------------------- inter_q mode ----------------------------
def default_inter_q_summary_path(args):
    ada = adaptive_tag(args.adaptive_budget)
    return (
        f"../result/{args.dataset}_{args.start}/{ada}/{args.strategy}/"
        f"{args.loss_type}/runner_inter_q_linear/runner_inter_q_linear_summary.pt"
    )


def run_inter_q_plot(args):
    summary_path = args.summary_path or default_inter_q_summary_path(args)
    summary = load_torch_file(summary_path)
    if not isinstance(summary, dict) or "results" not in summary:
        raise ValueError(f"Invalid summary format: {summary_path}")

    budget_items = sort_budget_items(summary["results"])
    if len(budget_items) == 0:
        raise ValueError(f"No valid budget entries found in summary['results']: {summary_path}")

    budgets = [x[0] for x in budget_items]
    baseline_ppl = [float(x[1]["baseline"]["ppl"]) for x in budget_items]
    inter_q_ppl = [float(x[1]["inter_q_linear"]["ppl"]) for x in budget_items]
    optimal_ppl = [float(x[1]["optimal"]["ppl"]) for x in budget_items]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(budgets, baseline_ppl, marker="o", linewidth=1.8, label="baseline qk routing")
    ax.plot(budgets, inter_q_ppl, marker="o", linewidth=1.8, linestyle="--", label="qWk routing")
    ax.plot(budgets, optimal_ppl, marker="o", linewidth=1.8, label="optimal routing")

    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = resolve_output_for_summary(args.output_path, summary_path, "ppl_vs_budget.png")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Loaded summary: {summary_path}")
    print(f"Saved plot: {output_path}")
    print("Budgets:", budgets)
    print("baseline qk routing:", baseline_ppl)
    print("qWk routing:", inter_q_ppl)
    print("optimal routing:", optimal_ppl)


# ---------------------------- inter_avgkv mode ----------------------------
def default_inter_avgkv_summary_path(args):
    ada = adaptive_tag(args.adaptive_budget)
    eval_start = args.start if args.eval_start is None else args.eval_start
    sample_tag = f"fit{args.start}" if eval_start == args.start else f"fit{args.start}_eval{eval_start}"
    return (
        f"../result/{args.dataset}_{sample_tag}/{ada}/{args.strategy}/"
        f"{args.loss_type}/runner_inter_avgKV/runner_inter_avgKV_summary.pt"
    )


def run_inter_avgkv_plot(args):
    summary_path = args.summary_path or default_inter_avgkv_summary_path(args)
    summary = load_torch_file(summary_path)
    if not isinstance(summary, dict) or "results" not in summary:
        raise ValueError(f"Invalid summary format: {summary_path}")

    budget_items = sort_budget_items(summary["results"])
    if len(budget_items) == 0:
        raise ValueError(f"No valid budget entries found in summary['results']: {summary_path}")

    budgets = [x[0] for x in budget_items]
    baseline_ppl = [float(x[1]["baseline"]["ppl"]) for x in budget_items if x[1].get("baseline") is not None]
    inter_avgkv_ppl = [float(x[1]["inter_avgkv"]["ppl"]) for x in budget_items]
    optimal_ppl = [float(x[1]["optimal"]["ppl"]) for x in budget_items if x[1].get("optimal") is not None]

    # Keep x-axis aligned when baseline/optimal is partially missing.
    b_base = [x[0] for x in budget_items if x[1].get("baseline") is not None]
    b_opt = [x[0] for x in budget_items if x[1].get("optimal") is not None]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if len(baseline_ppl) > 0:
        ax.plot(b_base, baseline_ppl, marker="o", linewidth=1.8, label="baseline qk routing")
    ax.plot(budgets, inter_avgkv_ppl, marker="o", linewidth=1.8, linestyle="--", label="inter avgKV routing")
    if len(optimal_ppl) > 0:
        ax.plot(b_opt, optimal_ppl, marker="o", linewidth=1.8, label="optimal routing")

    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = resolve_output_for_summary(args.output_path, summary_path, "ppl_vs_budget_avgkv.png")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Loaded summary: {summary_path}")
    print(f"Saved plot: {output_path}")
    print("Budgets:", budgets)
    print("baseline qk routing:", baseline_ppl)
    print("inter avgKV routing:", inter_avgkv_ppl)
    print("optimal routing:", optimal_ppl)


# ---------------------------- inter_q_eval mode ----------------------------
def default_inter_q_eval_summary_path(args):
    ada = adaptive_tag(args.adaptive_budget)
    eval_start = args.fit_start if args.eval_start is None else args.eval_start
    sample_tag = f"fit{args.fit_start}" if eval_start == args.fit_start else f"fit{args.fit_start}_eval{eval_start}"
    runner_name = "runner_inter_q_linear" if args.variant == "linear" else "runner_inter_q_bias"
    return (
        f"../result/{args.dataset}_{sample_tag}/{ada}/{args.strategy}/"
        f"{args.loss_type}/{runner_name}/{runner_name}_summary.pt"
    )


def resolve_inter_q_eval_key_and_label(variant):
    if variant == "linear":
        return "inter_q_linear", "eval qWk routing"
    if variant == "bias":
        return "inter_q_bias", "eval q+bias routing"
    raise ValueError(f"Unknown variant: {variant}")


def pick_metric(entry, key):
    metric = entry.get(key, None)
    if metric is None or not isinstance(metric, dict):
        return None
    if "ppl" not in metric:
        return None
    return float(metric["ppl"])


def collect_series(budget_items, key):
    budgets, values = [], []
    for b, entry in budget_items:
        v = pick_metric(entry, key)
        if v is None:
            continue
        budgets.append(float(b))
        values.append(float(v))
    return budgets, values


def run_inter_q_eval_plot(args):
    if args.eval_start is None:
        args.eval_start = args.fit_start

    summary_path = args.summary_path or default_inter_q_eval_summary_path(args)
    summary = load_torch_file(summary_path)
    if not isinstance(summary, dict) or "results" not in summary:
        raise ValueError(f"Invalid summary format: {summary_path}")

    budget_items = sort_budget_items(summary["results"])
    if len(budget_items) == 0:
        raise ValueError(f"No valid budget entries found in summary['results']: {summary_path}")

    inter_key, inter_label = resolve_inter_q_eval_key_and_label(args.variant)

    b_base, y_base = collect_series(budget_items, "baseline")
    b_inter, y_inter = collect_series(budget_items, inter_key)
    b_opt, y_opt = collect_series(budget_items, "optimal")

    if len(y_inter) == 0:
        raise ValueError(f"No {inter_key} routing ppl found in summary['results']")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if len(y_base) > 0:
        ax.plot(b_base, y_base, marker="o", linewidth=1.8, label="eval baseline routing")
    ax.plot(b_inter, y_inter, marker="o", linewidth=1.8, linestyle="--", label=inter_label)
    if len(y_opt) > 0:
        ax.plot(b_opt, y_opt, marker="o", linewidth=1.8, label="eval optimal routing")

    ax.set_xlabel("budget")
    ax.set_ylabel("PPL")
    ax.set_title("Eval PPL vs budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    suffix = "qwk" if args.variant == "linear" else "qbias"
    output_path = resolve_output_for_summary(args.output_path, summary_path, f"eval_ppl_vs_budget_{suffix}.png")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Loaded summary: {summary_path}")
    print(f"Saved plot: {output_path}")


def main():
    args = parse_args()

    if args.plot_kind == "routing_gap":
        run_routing_gap_plot(args)
    elif args.plot_kind == "inter":
        run_inter_plot(args)
    elif args.plot_kind == "inter_q":
        run_inter_q_plot(args)
    elif args.plot_kind == "inter_q_eval":
        run_inter_q_eval_plot(args)
    elif args.plot_kind == "inter_avgkv":
        run_inter_avgkv_plot(args)
    else:
        raise ValueError(f"Unsupported --plot-kind: {args.plot_kind}")


if __name__ == "__main__":
    main()
