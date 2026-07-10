"""
Summarize mean metrics from multiple compare scripts and draw one line chart.

Methods included:
- compare_count_sumV
- compare_count_avgKV
- compare_count_oracle
- compare_count_oracle_all
- compare_q_linear

Reads each stats .pt file, extracts:
- mean_base_metric
- method-specific mean metric
Then plots both lines over method index.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch


METHOD_SPECS = [
    {
        "name": "sumV",
        "compare_tag": "compare_count_sumV",
        "stats_file": "compare_count_sumV_stats.pt",
        "metric_key": "mean_sumv_metric",
    },
    {
        "name": "avgKV",
        "compare_tag": "compare_count_avgKV",
        "stats_file": "compare_count_avgKV_stats.pt",
        "metric_key": "mean_avgkv_metric",
    },
    {
        "name": "oracle",
        "compare_tag": "compare_count_oracle",
        "stats_file": "compare_count_oracle_stats.pt",
        "metric_key": "mean_oracle_metric",
    },
    {
        "name": "oracle_all",
        "compare_tag": "compare_count_oracle_all",
        "stats_file": "compare_count_oracle_all_stats.pt",
        "metric_key": "mean_oracle_all_metric",
    },
    {
        "name": "q_linear",
        "compare_tag": "compare_q_linear",
        "stats_file": "compare_q_linear_stats.pt",
        "metric_key": "mean_linear_metric",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read mean metrics from compare stats and draw one summary line chart."
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True)

    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="h2o")
    parser.add_argument("--loss-type", type=str, default="v_l2")
    parser.add_argument("--adaptive-budget", action="store_true")

    parser.add_argument(
        "--head-tag",
        type=str,
        default=None,
        help="Optional layer-head folder name, e.g. layer15_heads_32 or layer15_head3.",
    )
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)

    return parser.parse_args()


def _find_stats_path(root_dir, compare_tag, stats_file, layer, budget, head_tag=None):
    budget_dir = f"budget_{budget:g}"
    if head_tag is not None:
        path = os.path.join(root_dir, compare_tag, head_tag, budget_dir, stats_file)
        return path if os.path.exists(path) else None

    pattern = os.path.join(
        root_dir,
        compare_tag,
        f"layer{layer}_*",
        budget_dir,
        stats_file,
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matched stats for {compare_tag}: {matches}. "
            "Please set --head-tag explicitly."
        )
    return matches[0]


def _layer_head_dir_from_stats_path(stats_path):
    budget_dir = os.path.dirname(stats_path)
    layer_head_dir = os.path.basename(os.path.dirname(budget_dir))
    return layer_head_dir


def _load_mean_metrics(stats_path, metric_key):
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if "mean_base_metric" not in stats:
        raise KeyError(f"mean_base_metric not found in {stats_path}")
    if metric_key not in stats:
        raise KeyError(f"{metric_key} not found in {stats_path}")
    return float(stats["mean_base_metric"]), float(stats[metric_key])


def main():
    args = parse_args()

    adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
    root_dir = os.path.join(
        "..",
        "result",
        f"{args.dataset}_{args.start}",
        adaptive_str,
        args.strategy,
        args.loss_type,
    )

    first_spec = METHOD_SPECS[0]
    first_path = _find_stats_path(
        root_dir=root_dir,
        compare_tag=first_spec["compare_tag"],
        stats_file=first_spec["stats_file"],
        layer=args.layer,
        budget=args.budget,
        head_tag=args.head_tag,
    )
    if first_path is None:
        raise FileNotFoundError(
            f"Cannot find {first_spec['compare_tag']} stats under {root_dir} "
            f"for layer={args.layer}, budget={args.budget:g}."
        )

    layer_head_dir = args.head_tag or _layer_head_dir_from_stats_path(first_path)

    names = []
    base_vals = []
    method_vals = []

    used_paths = {}
    for spec in METHOD_SPECS:
        stats_path = _find_stats_path(
            root_dir=root_dir,
            compare_tag=spec["compare_tag"],
            stats_file=spec["stats_file"],
            layer=args.layer,
            budget=args.budget,
            head_tag=layer_head_dir,
        )
        if stats_path is None:
            raise FileNotFoundError(
                f"Missing {spec['compare_tag']} stats for head dir {layer_head_dir}. "
                f"Expected file: {os.path.join(root_dir, spec['compare_tag'], layer_head_dir, f'budget_{args.budget:g}', spec['stats_file'])}"
            )

        mean_base, mean_method = _load_mean_metrics(stats_path, spec["metric_key"])

        names.append(spec["name"])
        base_vals.append(mean_base)
        method_vals.append(mean_method)
        used_paths[spec["name"]] = stats_path

    base_ref = base_vals[0]
    max_base_delta = max(abs(x - base_ref) for x in base_vals)

    if args.output_dir is None:
        out_dir = os.path.join(
            root_dir,
            "compare_count_summary",
            layer_head_dir,
            f"budget_{args.budget:g}",
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(out_dir, "mean_metrics.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("method\tmean_base_metric\tmean_method_metric\tdelta_base_minus_method\n")
        for name, b, m in zip(names, base_vals, method_vals):
            f.write(f"{name}\t{b:.8e}\t{m:.8e}\t{(b - m):.8e}\n")

    x = list(range(len(names)))
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(x, base_vals, marker="o", linewidth=1.5, label="mean_base_metric")
    plt.plot(x, method_vals, marker="o", linewidth=1.5, label="mean_method_metric")
    plt.xticks(x, names)
    plt.ylabel("mean v_l2")
    plt.xlabel("method")
    plt.title(
        f"Layer {args.layer}, budget={args.budget:g}, head={layer_head_dir}\n"
        f"max |base_i-base_0| = {max_base_delta:.3e}"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "mean_metrics.png")
    plt.savefig(plot_path, dpi=args.plot_dpi)
    plt.close()

    print("===== Compare Summary =====")
    print(f"root_dir={root_dir}")
    print(f"layer_head_dir={layer_head_dir}")
    print(f"layer={args.layer}, budget={args.budget:g}")
    print(f"max base metric difference={max_base_delta:.8e}")
    for name in names:
        print(f"[{name}] stats={used_paths[name]}")
    print(f"Saved mean table to: {tsv_path}")
    print(f"Saved line chart to: {plot_path}")


if __name__ == "__main__":
    main()
