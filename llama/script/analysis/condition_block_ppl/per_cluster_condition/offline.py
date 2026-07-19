"""Compare the additive per-cluster condition at a fixed exact-block count."""

import argparse
import json
import math
from pathlib import Path

import torch

from .scores import condition_scores, hybrid_certificates
from ..condition import _build_routing_pos_list, _resolve_query_positions
from ..delta_bound.term_ablation.metrics import topk_mask
from ..delta_bound.term_ablation.run import _layer_groups
from ...runtime import load_context
from ...runner_utils import set_seed, str_to_torch_dtype
from ...sanity import move_model_inputs_to_device


METHODS = ("original", "per_cluster", "term1", "mass_exp")
RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "per_cluster_condition"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument(
        "--delta-kinds",
        nargs="+",
        choices=("box", "oracle"),
        default=["box", "oracle"],
    )
    parser.add_argument("--sample-heads", type=int, default=4)
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--queries", type=int, nargs="+", default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--budget", type=float, default=0.0625, help=argparse.SUPPRESS)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--no-post-wo", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _median(values):
    return float(torch.tensor(list(values), dtype=torch.float64).median())


def _model_tag(model):
    return str(model).rstrip("/").split("/")[-1]


def _markdown(result):
    config = result["config"]
    lines = [
        "# Per-cluster condition offline result",
        "",
        f"Model `{config['model']}`, start `{config['start']}`, block size "
        f"`{config['block_size']}`.",
        "",
        "Ratios are matched-k hybrid errors relative to the original normalized score.",
        "",
    ]
    for layer, layer_data in result["layers"].items():
        lines.extend([f"## Layer {layer}", ""])
        for delta_kind, data in layer_data.items():
            diag = data["diagnostics"]
            lines.extend(
                [
                    f"### {delta_kind} delta",
                    "",
                    f"Median log10(S_delta): `{diag['median_log10_s_delta']:.3f}`; "
                    f"per-cluster term2 fraction: `{diag['median_per_cluster_term2_fraction']:.6f}`.",
                    "",
                    "| fraction | method | pre err/original | post-Wo err/original | overlap(original) |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for fraction, by_method in data["fractions"].items():
                for method in METHODS:
                    row = by_method[method]
                    lines.append(
                        f"| {float(fraction):.2f} | {method} | "
                        f"{row['pre_error_ratio_original']:.3f} | "
                        f"{row['post_error_ratio_original']:.3f} | "
                        f"{row['overlap_original']:.3f} |"
                    )
            cert = data["certificate"]
            lines.extend(
                [
                    "",
                    f"Per-cluster additive certificate violations: "
                    f"`{cert['violations']}/{cert['cases']}`; median error/certificate "
                    f"`{cert['median_error_over_certificate']:.6f}`.",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def run(args):
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if any(fraction <= 0.0 or fraction > 1.0 for fraction in args.fractions):
        raise ValueError("--fractions must lie in (0, 1]")
    set_seed(args.seed)
    args.budget = 1.0 / args.block_size
    if args.query_end is None:
        args.query_end = args.seq_len

    ctx = load_context(args, dtype=str_to_torch_dtype(args.dtype), device=args.device)
    ctx.model.eval()
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    positions = _resolve_query_positions(args, _build_routing_pos_list(args))
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "layers": {},
    }

    for layer in args.layers:
        print(f"[per-cluster] layer={layer}, block={args.block_size}", flush=True)
        layer_result = {}
        groups = list(_layer_groups(ctx, args, layer, positions, model_inputs))
        for delta_kind in args.delta_kinds:
            records = {
                str(fraction): {method: [] for method in METHODS}
                for fraction in args.fractions
            }
            diagnostics = []
            cert_ratios = []
            cert_violations = 0
            cert_cases = 0

            for group in groups:
                scores, parts = condition_scores(
                    group["p_hat"],
                    group[f"delta_{delta_kind}"],
                    group["b_c"],
                    group["b_all"],
                )
                total_pc = (parts["per_cluster_mass"] + parts["value"]).sum()
                diagnostics.append(
                    {
                        "log10_s_delta": math.log10(float(parts["s_delta"])),
                        "per_cluster_term2_fraction": float(
                            parts["value"].sum() / total_pc.clamp_min(1e-300)
                        ),
                    }
                )
                for fraction in args.fractions:
                    k = max(
                        1,
                        min(
                            group["state"].num_blocks,
                            int(round(fraction * group["state"].num_blocks)),
                        ),
                    )
                    masks = {
                        method: topk_mask(scores[method], k)
                        for method in METHODS
                    }
                    original_mask = masks["original"]
                    for method, selected in masks.items():
                        pre_error, post_error = group["state"].error(selected)
                        records[str(fraction)][method].append(
                            {
                                "pre_error": pre_error,
                                "post_error": post_error,
                                "overlap_original": float(
                                    (selected & original_mask).sum()
                                ) / k,
                            }
                        )
                    selected = masks["per_cluster"]
                    _tight, additive = hybrid_certificates(
                        group["p_hat"],
                        group[f"delta_{delta_kind}"],
                        group["b_c"],
                        group["b_all"],
                        ~selected,
                    )
                    error = records[str(fraction)]["per_cluster"][-1]["pre_error"]
                    cert_cases += 1
                    cert_violations += error > additive + 1e-9
                    cert_ratios.append(error / max(additive, 1e-300))

            fractions = {}
            for fraction, by_method in records.items():
                original_pre = _mean(row["pre_error"] for row in by_method["original"])
                original_post = _mean(row["post_error"] for row in by_method["original"])
                fractions[fraction] = {}
                for method, rows in by_method.items():
                    mean_pre = _mean(row["pre_error"] for row in rows)
                    mean_post = _mean(row["post_error"] for row in rows)
                    fractions[fraction][method] = {
                        "pre_error": mean_pre,
                        "post_error": mean_post,
                        "pre_error_ratio_original": mean_pre / original_pre,
                        "post_error_ratio_original": mean_post / original_post,
                        "overlap_original": _mean(
                            row["overlap_original"] for row in rows
                        ),
                    }
            layer_result[delta_kind] = {
                "groups": len(groups),
                "diagnostics": {
                    "median_log10_s_delta": _median(
                        row["log10_s_delta"] for row in diagnostics
                    ),
                    "median_per_cluster_term2_fraction": _median(
                        row["per_cluster_term2_fraction"] for row in diagnostics
                    ),
                },
                "fractions": fractions,
                "certificate": {
                    "violations": cert_violations,
                    "cases": cert_cases,
                    "median_error_over_certificate": _median(cert_ratios),
                    "max_error_over_certificate": max(cert_ratios),
                },
            }
        result["layers"][str(layer)] = layer_result

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            RESULT_ROOT
            / _model_tag(args.model)
            / f"{args.dataset}_{args.start}"
            / f"block_{args.block_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    markdown_path.write_text(_markdown(result))
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")
    return result


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
