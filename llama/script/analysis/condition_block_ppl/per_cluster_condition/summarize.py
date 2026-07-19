"""Aggregate per-cluster-condition offline results across dataset windows."""

import argparse
import json
import math
from pathlib import Path


METHODS = ("original", "per_cluster", "term1", "mass_exp")
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "per_cluster_condition"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--model-tag", default="Llama-2-7b-hf")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument(
        "--starts", type=int, nargs="+", default=[0, 1024, 2048, 3072, 4096]
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _interval(values):
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "ci95": [mean, mean]}
    std = math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    half = T95.get(len(values), 1.96) * std / math.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def _window_ratio(data, delta_kind, fraction, method, key):
    method_total = 0.0
    original_total = 0.0
    overlap = 0.0
    for layer in data["layers"].values():
        rows = layer[delta_kind]["fractions"][str(fraction)]
        method_total += float(rows[method][key])
        original_total += float(rows["original"][key])
        overlap += float(rows[method]["overlap_original"])
    return method_total / original_total, overlap / len(data["layers"])


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
            raise FileNotFoundError(path)
        windows[str(start)] = json.loads(path.read_text())

    reference = next(iter(windows.values()))
    fractions = [str(value) for value in reference["config"]["fractions"]]
    delta_kinds = reference["config"]["delta_kinds"]
    result = {
        "config": {
            "model_tag": args.model_tag,
            "dataset": args.dataset,
            "starts": args.starts,
            "block_size": args.block_size,
            "layers": reference["config"]["layers"],
            "fractions": [float(value) for value in fractions],
            "delta_kinds": delta_kinds,
        },
        "results": {},
        "diagnostics": {},
    }
    for delta_kind in delta_kinds:
        result["results"][delta_kind] = {}
        result["diagnostics"][delta_kind] = {}
        log_s = []
        value_fractions = []
        violations = cases = 0
        for data in windows.values():
            layer_rows = [layer[delta_kind] for layer in data["layers"].values()]
            log_s.append(
                sum(row["diagnostics"]["median_log10_s_delta"] for row in layer_rows)
                / len(layer_rows)
            )
            value_fractions.append(
                sum(
                    row["diagnostics"]["median_per_cluster_term2_fraction"]
                    for row in layer_rows
                )
                / len(layer_rows)
            )
            violations += sum(row["certificate"]["violations"] for row in layer_rows)
            cases += sum(row["certificate"]["cases"] for row in layer_rows)
        result["diagnostics"][delta_kind] = {
            "mean_log10_s_delta": sum(log_s) / len(log_s),
            "mean_per_cluster_term2_fraction": (
                sum(value_fractions) / len(value_fractions)
            ),
            "certificate_violations": violations,
            "certificate_cases": cases,
        }

        for fraction in fractions:
            result["results"][delta_kind][fraction] = {}
            for method in METHODS:
                pre = []
                post = []
                overlaps = []
                for data in windows.values():
                    pre_ratio, overlap = _window_ratio(
                        data, delta_kind, fraction, method, "pre_error"
                    )
                    post_ratio, _ = _window_ratio(
                        data, delta_kind, fraction, method, "post_error"
                    )
                    pre.append(pre_ratio)
                    post.append(post_ratio)
                    overlaps.append(overlap)
                result["results"][delta_kind][fraction][method] = {
                    "pre_error_ratio_original": _interval(pre),
                    "post_error_ratio_original": _interval(post),
                    "overlap_original_mean": sum(overlaps) / len(overlaps),
                    "window_pre_ratios": pre,
                    "window_post_ratios": post,
                }
    return result


def _markdown(result):
    config = result["config"]
    lines = [
        "# Multi-window per-cluster-condition summary",
        "",
        f"Block size `{config['block_size']}`, starts `{config['starts']}`, layers "
        f"`{config['layers']}`. Intervals are 95% Student-t intervals over windows.",
        "",
    ]
    for delta_kind in config["delta_kinds"]:
        diag = result["diagnostics"][delta_kind]
        lines.extend(
            [
                f"## {delta_kind} delta",
                "",
                f"Mean log10(S_delta): `{diag['mean_log10_s_delta']:.3f}`; "
                f"per-cluster term2 fraction: "
                f"`{diag['mean_per_cluster_term2_fraction']:.8f}`; certificate "
                f"violations: `{diag['certificate_violations']}/"
                f"{diag['certificate_cases']}`.",
                "",
                "| fraction | method | pre err/original | post-Wo err/original | post 95% CI | overlap(original) |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for fraction in config["fractions"]:
            rows = result["results"][delta_kind][str(fraction)]
            for method in METHODS:
                row = rows[method]
                pre = row["pre_error_ratio_original"]["mean"]
                post = row["post_error_ratio_original"]["mean"]
                low, high = row["post_error_ratio_original"]["ci95"]
                lines.append(
                    f"| {fraction:.2f} | {method} | {pre:.3f} | {post:.3f} | "
                    f"[{low:.3f}, {high:.3f}] | {row['overlap_original_mean']:.3f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    result = aggregate(args)
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
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    markdown_path.write_text(_markdown(result))
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
