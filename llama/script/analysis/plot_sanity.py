import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch


VALID_METRICS = [
    "sanity_kl",
    "nll_gap",
    "teacher_nll",
    "student_nll",
    "nll_pair",
    "baseline_sanity_kl",
    "baseline_nll_gap",
    "baseline_teacher_nll",
    "baseline_student_nll",
    "baseline_nll_pair",
    "delta_sanity_kl",
    "delta_student_nll",
    "delta_nll_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sanity-check metric curves vs budget")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name in result path")
    parser.add_argument("--strategy", type=str, default="h2o", help="Strategy name in result path")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="logits_kl",
        choices=["logits_kl", "v_l2", "v_kl"],
        help="Loss type subdirectory to read from",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to plot",
    )
    layer_group.add_argument(
        "--all-layers",
        action="store_true",
        help="Plot all layers that have result files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sanity_kl",
        choices=VALID_METRICS,
        help="Sanity metric to plot on y-axis",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional output image path; if empty, save under llama/result/plots",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale for y-axis metric",
    )
    return parser.parse_args()


def normalize_entry(entry):
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, (tuple, list)):
        return {"opt": entry}
    return {}


def load_layer_metric(result_path, metric_name):
    if not result_path.exists():
        return [], []

    result = torch.load(result_path, weights_only=False)
    budgets = []
    values = []

    for budget, raw_entry in result.items():
        entry = normalize_entry(raw_entry)
        if metric_name in entry:
            budgets.append(float(budget))
            values.append(float(entry[metric_name]))

    if not budgets:
        return [], []

    pairs = sorted(zip(budgets, values), key=lambda x: x[0])
    sorted_budgets = [p[0] for p in pairs]
    sorted_values = [p[1] for p in pairs]
    return sorted_budgets, sorted_values


def load_layer_metric_pair(result_path, metric_a, metric_b):
    if not result_path.exists():
        return [], [], []

    result = torch.load(result_path, weights_only=False)
    budgets = []
    values_a = []
    values_b = []

    for budget, raw_entry in result.items():
        entry = normalize_entry(raw_entry)
        if metric_a in entry and metric_b in entry:
            budgets.append(float(budget))
            values_a.append(float(entry[metric_a]))
            values_b.append(float(entry[metric_b]))

    if not budgets:
        return [], [], []

    pairs = sorted(zip(budgets, values_a, values_b), key=lambda x: x[0])
    sorted_budgets = [p[0] for p in pairs]
    sorted_a = [p[1] for p in pairs]
    sorted_b = [p[2] for p in pairs]
    return sorted_budgets, sorted_a, sorted_b


def default_save_path(metric, dataset, strategy, loss_type):
    llama_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = llama_dir / "result" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sanity_{metric}_{dataset}_{strategy}_{loss_type}.png"


def discover_available_layers(result_root, dataset, strategy, loss_type):
    layer_indices = []
    for pt_path in result_root.glob(f"layer*/{dataset}/{strategy}/{loss_type}/result.pt"):
        layer_dir_name = pt_path.parents[3].name
        if not layer_dir_name.startswith("layer"):
            continue
        suffix = layer_dir_name[len("layer"):]
        if suffix.isdigit():
            layer_indices.append(int(suffix))
    return sorted(set(layer_indices))


def main():
    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent.parent
    llama_dir = script_dir.parent
    result_root = llama_dir / "result"

    if args.all_layers:
        target_layers = discover_available_layers(
            result_root,
            args.dataset,
            args.strategy,
            args.loss_type,
        )
        if not target_layers:
            raise RuntimeError(
                "--all-layers was set but no layer result files were found under "
                f"{result_root} for ({args.dataset}, {args.strategy}, {args.loss_type})."
            )
    elif args.layers is None:
        target_layers = [5, 10, 15, 20, 25, 30]
    else:
        target_layers = args.layers

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for layer_idx in target_layers:
        result_path = (
            result_root
            / f"layer{layer_idx}"
            / args.dataset
            / args.strategy
            / args.loss_type
            / "result.pt"
        )
        if args.metric in ["nll_pair", "baseline_nll_pair"]:
            metric_a = "teacher_nll" if args.metric == "nll_pair" else "baseline_teacher_nll"
            metric_b = "student_nll" if args.metric == "nll_pair" else "baseline_student_nll"
            budgets, teacher_vals, student_vals = load_layer_metric_pair(
                result_path,
                metric_a,
                metric_b,
            )
            if not budgets:
                print(
                    f"[WARN] No paired NLL metrics found for layer {layer_idx}: {result_path}"
                )
                continue

            plt.plot(
                budgets,
                teacher_vals,
                marker="o",
                linewidth=2,
                linestyle="--",
                label=f"layer {layer_idx} {metric_a}",
            )
            plt.plot(
                budgets,
                student_vals,
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"layer {layer_idx} {metric_b}",
            )
            plotted_any = True
        else:
            budgets, values = load_layer_metric(result_path, args.metric)
            if not budgets:
                print(f"[WARN] No metric '{args.metric}' found for layer {layer_idx}: {result_path}")
                continue

            plt.plot(budgets, values, marker="o", linewidth=2, label=f"layer {layer_idx}")
            plotted_any = True

    if not plotted_any:
        raise RuntimeError(
            f"No plottable data found for metric '{args.metric}'. "
            "Please run sanity check first to populate metrics in result.pt."
        )

    plt.xscale("log")
    if args.logy:
        plt.yscale("log")

    plt.xlabel("budget")
    ylabel = "NLL" if args.metric in ["nll_pair", "baseline_nll_pair"] else args.metric
    plt.ylabel(ylabel)
    plt.title(
        f"Sanity check curve: {args.metric} vs budget "
        f"({args.dataset}, {args.strategy}, {args.loss_type})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = (
        Path(args.save_path)
        if args.save_path
        else default_save_path(args.metric, args.dataset, args.strategy, args.loss_type)
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()
