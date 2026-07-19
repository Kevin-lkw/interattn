"""Aggregate term-ablation summaries across independent dataset windows."""

import argparse
import json
import math
from pathlib import Path


RESULT_ROOT = Path(__file__).resolve().parents[5] / "result" / "term_ablation"
DEFAULT_METHODS = (
    "full",
    "term1",
    "term2",
    "full_mass_exp",
    "full_t2_sat",
    "full_t2_size_gate",
    "full_t2_clip",
    "mix_0.25",
    "mix_0.5",
    "mix_0.75",
    "mass_exp_mix_0.25",
    "mass_exp_mix_0.5",
    "mass_exp_mix_0.75",
)
T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--model-tag", default="Llama-2-7b-hf")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--starts", type=int, nargs="+", default=[0, 1024, 2048, 3072, 4096])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--delta-kinds", nargs="+", choices=("box", "oracle"), default=["box", "oracle"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _mean_interval(values):
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean]}
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    critical = T95.get(n, 1.96)
    half = critical * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def _window_ratio(data, layers, delta_kind, method, fraction, key):
    method_sum = 0.0
    full_sum = 0.0
    agreement = 0.0
    for layer in layers:
        methods = data["layers"][str(layer)]["methods"][delta_kind]
        row = methods[method]["fractions"][str(fraction)]
        base = methods["full"]["fractions"][str(fraction)]
        method_sum += row[key]
        full_sum += base[key]
        agreement += row["agreement_full"]
    return method_sum / full_sum, agreement / len(layers)


def aggregate(args):
    windows = {}
    for start in args.starts:
        path = (
            args.result_root
            / args.model_tag
            / f"{args.dataset}_{start}"
            / f"block_{args.block_size}"
            / "summary.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing run summary: {path}")
        windows[str(start)] = json.loads(path.read_text())

    output = {
        "config": {
            "model_tag": args.model_tag,
            "dataset": args.dataset,
            "starts": args.starts,
            "block_size": args.block_size,
            "layers": args.layers,
            "fractions": args.fractions,
            "delta_kinds": args.delta_kinds,
            "methods": args.methods,
        },
        "results": {},
    }
    for delta_kind in args.delta_kinds:
        output["results"][delta_kind] = {}
        for method in args.methods:
            output["results"][delta_kind][method] = {}
            for fraction in args.fractions:
                pre_ratios = []
                post_ratios = []
                agreements = []
                for data in windows.values():
                    pre, agreement = _window_ratio(
                        data, args.layers, delta_kind, method, fraction, "pre_error"
                    )
                    post, _ = _window_ratio(
                        data, args.layers, delta_kind, method, fraction, "post_error"
                    )
                    pre_ratios.append(pre)
                    post_ratios.append(post)
                    agreements.append(agreement)
                output["results"][delta_kind][method][str(fraction)] = {
                    "pre_error_ratio_full": _mean_interval(pre_ratios),
                    "post_error_ratio_full": _mean_interval(post_ratios),
                    "agreement_full_mean": sum(agreements) / len(agreements),
                    "window_pre_ratios": pre_ratios,
                    "window_post_ratios": post_ratios,
                }
    return output


def _markdown(summary):
    config = summary["config"]
    lines = [
        "# Multi-window term-ablation summary",
        "",
        f"Block size `{config['block_size']}`, starts `{config['starts']}`, "
        f"layers `{config['layers']}`.",
        "",
        "Ratios are computed against the original full score within each window. "
        "Intervals are 95% Student-t intervals over window-level ratios.",
        "",
    ]
    for delta_kind in config["delta_kinds"]:
        lines.extend([f"## {delta_kind} delta", ""])
        for fraction in config["fractions"]:
            lines.extend(
                [
                    f"### Fraction {fraction:g}",
                    "",
                    "| method | pre err/full | post-Wo err/full | post 95% CI | agree(full) |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in config["methods"]:
                row = summary["results"][delta_kind][method][str(fraction)]
                pre = row["pre_error_ratio_full"]["mean"]
                post = row["post_error_ratio_full"]["mean"]
                low, high = row["post_error_ratio_full"]["ci95"]
                lines.append(
                    f"| {method} | {pre:.3f} | {post:.3f} | [{low:.3f}, {high:.3f}] | "
                    f"{row['agreement_full_mean']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = aggregate(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.result_root
            / args.model_tag
            / f"{args.dataset}_multiwindow"
            / f"block_{args.block_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path.write_text(_markdown(summary))
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
