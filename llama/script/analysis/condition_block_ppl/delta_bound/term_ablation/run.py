"""Measure which condition term actually improves matched-budget block selection."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from .metrics import (
    HybridState,
    finite_mean,
    finite_median,
    spearman,
    topk_mask,
    topk_metrics,
)
from ...condition import _build_routing_pos_list, _choose_evenly, _resolve_query_positions
from ...condition_block_corr import _range_bound_delta
from ...post_wo_experiment.core import split_o_proj_weight
from ....experiment_utils import resolve_head_indices, validate_common_args
from ....online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ....runtime import load_context
from ....runner_utils import set_seed, str_to_torch_dtype
from ....sanity import grouped_query_heads, move_model_inputs_to_device


RESULT_ROOT = Path(__file__).resolve().parents[5] / "result" / "term_ablation"
DEFAULT_METHODS = (
    "full",
    "term1",
    "term2",
    "term1_log_exact",
    "term1_log_exp",
    "term2_sat",
    "full_no_minus1",
    "full_mass_exp",
    "full_t2_sat",
    "full_t2_size_gate",
    "full_t2_clip",
    "simple_full",
    "simple_full_size_gate",
    "simple_full_clip",
    "mix_0.25",
    "mix_0.5",
    "mix_0.75",
    "mass_exp_mix_0.25",
    "mass_exp_mix_0.5",
    "mass_exp_mix_0.75",
    "mix_1.5",
    "mix_2",
    "p_hat",
    "p_hat_b_c",
    "delta",
)


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
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--delta-kinds", nargs="+", choices=("box", "oracle"), default=["box", "oracle"])
    parser.add_argument("--sample-heads", type=int, default=4)
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--queries", type=int, nargs="+", default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--budget", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--greedy-oracle", action="store_true")
    parser.add_argument("--no-post-wo", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _model_tag(model):
    return str(model).rstrip("/").split("/")[-1]


def _log_cosh_minus_one(delta):
    half = delta / 2.0
    return math.log(2.0) + 2.0 * torch.log(torch.sinh(half))


def compute_scores(p_hat, delta, b_c, b_all, sizes=None):
    """Return exact terms, deployable simplifications, and diagnostic controls."""
    delta = delta.double().clamp(min=0.0, max=80.0)
    p_hat = p_hat.double()
    b_c = b_c.double()
    f_value = torch.cosh(delta)
    normalizer = (p_hat * f_value).sum().clamp_min(1e-300)
    term1 = p_hat * 2.0 * b_all * (f_value - 1.0) / normalizer
    term2 = p_hat * 2.0 * b_c * torch.tanh(delta / 2.0)

    log_p = torch.log(p_hat.clamp_min(1e-300))
    mass_asymptotic = torch.softmax(log_p + delta, dim=0)
    term1_mass_exp = 2.0 * b_all * mass_asymptotic
    if sizes is None:
        non_singleton = torch.ones_like(delta)
    else:
        non_singleton = (sizes.to(delta.device) > 1).double()
    tanh_clip = (delta / 2.0).clamp(max=1.0)
    scores = {
        "full": term1 + term2,
        "term1": term1,
        "term2": term2,
        # Exactly the same ranking as term1; removes B and the global S normalizer.
        "term1_log_exact": log_p + _log_cosh_minus_one(delta),
        # Large-delta approximation: cosh(delta)-1 is proportional to exp(delta).
        "term1_log_exp": log_p + delta,
        # Saturated-tanh approximation; the factor 2 is irrelevant for ranking.
        "term2_sat": p_hat * b_c,
        "full_no_minus1": p_hat * 2.0 * b_all * f_value / normalizer + term2,
        # Simplify only the mass channel; keep low-delta behavior in term2 exact.
        "full_mass_exp": term1_mass_exp + term2,
        "full_t2_sat": term1 + 2.0 * p_hat * b_c,
        "full_t2_size_gate": term1 + 2.0 * p_hat * b_c * non_singleton,
        "full_t2_clip": term1 + 2.0 * p_hat * b_c * tanh_clip,
        "simple_full": term1_mass_exp + 2.0 * p_hat * b_c,
        "simple_full_size_gate": (
            term1_mass_exp + 2.0 * p_hat * b_c * non_singleton
        ),
        "simple_full_clip": (
            term1_mass_exp + 2.0 * p_hat * b_c * tanh_clip
        ),
        "mix_0.25": term1 + 0.25 * term2,
        "mix_0.5": term1 + 0.5 * term2,
        "mix_0.75": term1 + 0.75 * term2,
        "mass_exp_mix_0.25": term1_mass_exp + 0.25 * term2,
        "mass_exp_mix_0.5": term1_mass_exp + 0.5 * term2,
        "mass_exp_mix_0.75": term1_mass_exp + 0.75 * term2,
        "mix_1.5": term1 + 1.5 * term2,
        "mix_2": term1 + 2.0 * term2,
        "p_hat": p_hat,
        "p_hat_b_c": p_hat * b_c,
        "delta": delta,
    }
    diagnostics = {
        "log10_S": math.log10(float(normalizer)),
        "tanh_saturation": float((torch.tanh(delta / 2.0) > 0.99).double().mean()),
        "term1_total_fraction": float(term1.sum() / (term1 + term2).sum().clamp_min(1e-300)),
    }
    return scores, diagnostics


def _build_group(q, k_head, v_head, query_pos, block_size, post_gram):
    total = int(query_pos) + 1
    qf = q.float()
    visible_k = k_head[:total].float()
    visible_v = v_head[:total].float()
    scale = math.sqrt(qf.numel())
    b_all = float(torch.linalg.vector_norm(visible_v, dim=-1).max())

    rows = []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        kc = visible_k[start:end]
        vc = visible_v[start:end]
        score = torch.mv(kc, qf) / scale
        s_c = torch.dot(qf, kc.mean(dim=0)) / scale
        rows.append(
            {
                "size": end - start,
                "z_hat": math.log(end - start) + float(s_c),
                "z_true": float(torch.logsumexp(score, dim=0)),
                "u_true": (torch.softmax(score, dim=0).unsqueeze(-1) * vc).sum(dim=0),
                "v_bar": vc.mean(dim=0),
                "b_c": float(torch.linalg.vector_norm(vc, dim=-1).max()),
                "delta_box": float(_range_bound_delta(qf, kc, s_c, scale)),
                "delta_oracle": float((score - score.mean()).abs().max()),
            }
        )

    z_hat = torch.tensor([row["z_hat"] for row in rows], device=q.device, dtype=torch.float64)
    z_true = torch.tensor([row["z_true"] for row in rows], device=q.device, dtype=torch.float64)
    shift = torch.maximum(z_hat.max(), z_true.max())
    approx_den = torch.exp(z_hat - shift)
    exact_den = torch.exp(z_true - shift)
    v_bar = torch.stack([row["v_bar"] for row in rows]).double()
    u_true = torch.stack([row["u_true"] for row in rows]).double()
    approx_num = approx_den[:, None] * v_bar
    exact_num = exact_den[:, None] * u_true
    full_output = exact_num.sum(dim=0) / exact_den.sum()
    state = HybridState(
        approx_num=approx_num,
        approx_den=approx_den,
        exact_num=exact_num,
        exact_den=exact_den,
        full_output=full_output,
        post_gram=None if post_gram is None else post_gram.double(),
    )

    exact_contrib = exact_num / exact_den.sum()
    approx_contrib = approx_num / approx_den.sum()
    contribution_delta = exact_contrib - approx_contrib
    local_pre = torch.linalg.vector_norm(contribution_delta, dim=-1)
    if post_gram is None:
        local_post = torch.full_like(local_pre, float("nan"))
    else:
        local_post = torch.einsum(
            "bd,de,be->b", contribution_delta, post_gram.double(), contribution_delta
        ).clamp_min(0.0).sqrt()

    return {
        "state": state,
        "p_hat": approx_den / approx_den.sum(),
        "b_c": torch.tensor([row["b_c"] for row in rows], device=q.device, dtype=torch.float64),
        "b_all": b_all,
        "delta_box": torch.tensor([row["delta_box"] for row in rows], device=q.device, dtype=torch.float64),
        "delta_oracle": torch.tensor([row["delta_oracle"] for row in rows], device=q.device, dtype=torch.float64),
        "sizes": torch.tensor([row["size"] for row in rows], device=q.device),
        "local_pre": local_pre,
        "local_post": local_post,
    }


def _layer_groups(ctx, args, layer_idx, positions, model_inputs):
    args.layer = layer_idx
    validate_common_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )
    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    if args.head is None and args.heads is None:
        head_idx = _choose_evenly(head_idx, args.sample_heads)
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=layer_idx,
        pos_list=positions,
        model_inputs=model_inputs,
        layer_to_patch={},
    )
    layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
    q_all = layer_ctx.rope_qkv[layer_idx]["q"].to(ctx.device)[0].float()
    k_all = layer_ctx.rope_qkv[layer_idx]["k"].to(ctx.device)[0].float()
    v_all = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0].float()
    layer = ctx.model.model.layers[layer_idx]
    w_heads = split_o_proj_weight(
        layer.self_attn.o_proj.weight,
        int(ctx.model_config.num_attention_heads),
        int(q_all.shape[-1]),
    ).float()

    count = 0
    for kv_head, _out, query_heads in grouped_query_heads(
        head_idx, ctx.model_config, num_kv_heads=k_all.shape[0]
    ):
        for query_head in query_heads:
            post_gram = None
            if not args.no_post_wo:
                w_head = w_heads[query_head]
                post_gram = w_head.T @ w_head
            for query_pos in positions:
                yield _build_group(
                    q_all[query_head, int(query_pos)],
                    k_all[kv_head],
                    v_all[kv_head],
                    int(query_pos),
                    args.block_size,
                    post_gram,
                )
                count += 1
                if args.max_groups is not None and count >= args.max_groups:
                    return


def _mean_metric(records, key):
    return finite_mean(record[key] for record in records)


def _relative_error_interval(records, base, key):
    paired = [
        (float(row[key]), float(ref[key]))
        for row, ref in zip(records, base)
        if math.isfinite(float(row[key])) and math.isfinite(float(ref[key]))
    ]
    if not paired:
        return {
            "ratio": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "win_rate": float("nan"),
        }
    values = torch.tensor(paired, dtype=torch.float64)
    mean_base = float(values[:, 1].mean())
    mean_diff = float((values[:, 0] - values[:, 1]).mean())
    if values.shape[0] > 1:
        sem_diff = float(
            (values[:, 0] - values[:, 1]).std(unbiased=True)
            / math.sqrt(values.shape[0])
        )
    else:
        sem_diff = 0.0
    return {
        "ratio": 1.0 + mean_diff / mean_base,
        "ci95_low": 1.0 + (mean_diff - 1.96 * sem_diff) / mean_base,
        "ci95_high": 1.0 + (mean_diff + 1.96 * sem_diff) / mean_base,
        "win_rate": float((values[:, 0] < values[:, 1]).double().mean()),
    }


def _summarize_layer(raw, fractions, delta_kinds):
    summary = {
        "groups": raw["groups"],
        "diagnostics": {},
        "methods": {},
        "greedy_oracle": {},
    }
    for delta_kind in delta_kinds:
        diag = raw["diagnostics"][delta_kind]
        summary["diagnostics"][delta_kind] = {
            key: finite_median(row[key] for row in diag)
            for key in ("log10_S", "tanh_saturation", "term1_total_fraction")
        }
        summary["methods"][delta_kind] = {}
        full_records = raw["records"][delta_kind]["full"]
        for method, records in raw["records"][delta_kind].items():
            rank_stats = raw["rank_stats"][delta_kind][method]
            method_out = {
                "spearman_local_pre_group_mean": _mean_metric(
                    rank_stats, "local_pre"
                ),
                "spearman_local_post_group_mean": _mean_metric(
                    rank_stats, "local_post"
                ),
                "spearman_gain_pre_group_mean": _mean_metric(
                    rank_stats, "gain_pre"
                ),
                "spearman_gain_post_group_mean": _mean_metric(
                    rank_stats, "gain_post"
                ),
                "spearman_local_pre_pooled": spearman(
                    torch.cat(raw["scores"][delta_kind][method]),
                    torch.cat(raw["local_pre"]),
                ),
                "spearman_local_post_pooled": spearman(
                    torch.cat(raw["scores"][delta_kind][method]),
                    torch.cat(raw["local_post"]),
                ),
                "spearman_gain_pre_pooled": spearman(
                    torch.cat(raw["scores"][delta_kind][method]),
                    torch.cat(raw["gain_pre"]),
                ),
                "spearman_gain_post_pooled": spearman(
                    torch.cat(raw["scores"][delta_kind][method]),
                    torch.cat(raw["gain_post"]),
                ),
                "fractions": {},
            }
            for frac in fractions:
                rows = records[str(frac)]
                base = full_records[str(frac)]
                mean_pre = _mean_metric(rows, "pre_error")
                mean_post = _mean_metric(rows, "post_error")
                base_pre = _mean_metric(base, "pre_error")
                base_post = _mean_metric(base, "post_error")
                pre_interval = _relative_error_interval(rows, base, "pre_error")
                post_interval = _relative_error_interval(rows, base, "post_error")
                method_out["fractions"][str(frac)] = {
                    key: _mean_metric(rows, key)
                    for key in (
                        "local_pre_overlap",
                        "local_pre_capture",
                        "local_pre_ndcg",
                        "local_post_overlap",
                        "local_post_capture",
                        "gain_pre_overlap",
                        "gain_pre_capture",
                        "gain_pre_ndcg",
                        "gain_post_overlap",
                        "gain_post_capture",
                        "pre_error",
                        "post_error",
                        "agreement_full",
                        "agreement_term1",
                        "agreement_term2",
                    )
                }
                method_out["fractions"][str(frac)].update(
                    {
                        "pre_error_ratio_full": mean_pre / base_pre,
                        "post_error_ratio_full": mean_post / base_post,
                        "pre_error_ratio_full_ci95": [
                            pre_interval["ci95_low"], pre_interval["ci95_high"]
                        ],
                        "post_error_ratio_full_ci95": [
                            post_interval["ci95_low"], post_interval["ci95_high"]
                        ],
                        "pre_error_win_rate_full": pre_interval["win_rate"],
                        "post_error_win_rate_full": post_interval["win_rate"],
                        "paired_pre_ratio_full_median": finite_median(
                            row["pre_error"] / max(ref["pre_error"], 1e-30)
                            for row, ref in zip(rows, base)
                        ),
                    }
                )
            summary["methods"][delta_kind][method] = method_out

    for objective, by_fraction in raw["greedy_oracle"].items():
        summary["greedy_oracle"][objective] = {
            frac: {
                "pre_error": _mean_metric(rows, "pre_error"),
                "post_error": _mean_metric(rows, "post_error"),
            }
            for frac, rows in by_fraction.items()
        }
    return summary


def _markdown(summary, args):
    lines = [
        "# Term-ablation result",
        "",
        f"Model: `{args.model}`; block size: `{args.block_size}`; fractions: `{args.fractions}`.",
        "",
        "`err/full` is the ratio of mean matched-k hybrid error to the original full score. Lower is better.",
        "",
    ]
    show_methods = [method for method in DEFAULT_METHODS if method not in {"p_hat", "p_hat_b_c", "delta"}]
    focus_frac = min(args.fractions, key=lambda value: abs(value - 0.1))
    for layer, layer_data in summary["layers"].items():
        lines.extend([f"## Layer {layer}", ""])
        for delta_kind in args.delta_kinds:
            diag = layer_data["diagnostics"][delta_kind]
            lines.extend(
                [
                    f"### {delta_kind} delta",
                    "",
                    f"Median log10(S)={diag['log10_S']:.3f}, tanh saturation={diag['tanh_saturation']:.3f}, "
                    f"term1 total fraction={diag['term1_total_fraction']:.3f}.",
                    "",
                    f"Metrics at fraction {focus_frac:g}:",
                    "",
                    "| method | Sp(local) | Sp(gain) | overlap(local) | capture(gain) | pre err/full | post-Wo err/full | agree(full) |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            methods = layer_data["methods"][delta_kind]
            for method in show_methods:
                row = methods[method]
                frac = row["fractions"][str(focus_frac)]
                lines.append(
                    f"| {method} | {row['spearman_local_pre_group_mean']:.3f} | "
                    f"{row['spearman_gain_pre_group_mean']:.3f} | "
                    f"{frac['local_pre_overlap']:.3f} | {frac['gain_pre_capture']:.3f} | "
                    f"{frac['pre_error_ratio_full']:.3f} | {frac['post_error_ratio_full']:.3f} | "
                    f"{frac['agreement_full']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def run(args):
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if not args.fractions or any(frac <= 0.0 or frac > 1.0 for frac in args.fractions):
        raise ValueError("--fractions must lie in (0, 1]")
    set_seed(args.seed)
    args.budget = 1.0 / args.block_size
    if args.query_end is None:
        args.query_end = args.seq_len
    ctx = load_context(args, dtype=str_to_torch_dtype(args.dtype), device=args.device)
    ctx.model.eval()
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    positions = _resolve_query_positions(args, _build_routing_pos_list(args))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            RESULT_ROOT
            / _model_tag(args.model)
            / f"{args.dataset}_{args.start}"
            / f"block_{args.block_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "positions": [int(pos) for pos in positions],
        "layers": {},
    }
    for layer_idx in args.layers:
        print(f"[term-ablation] layer={layer_idx}, block={args.block_size}", flush=True)
        raw = {
            "groups": 0,
            "diagnostics": {kind: [] for kind in args.delta_kinds},
            "scores": {
                kind: {method: [] for method in DEFAULT_METHODS}
                for kind in args.delta_kinds
            },
            "records": {
                kind: {
                    method: {str(frac): [] for frac in args.fractions}
                    for method in DEFAULT_METHODS
                }
                for kind in args.delta_kinds
            },
            "rank_stats": {
                kind: {method: [] for method in DEFAULT_METHODS}
                for kind in args.delta_kinds
            },
            "local_pre": [],
            "local_post": [],
            "gain_pre": [],
            "gain_post": [],
            "greedy_oracle": defaultdict(
                lambda: {str(frac): [] for frac in args.fractions}
            ),
        }

        for group in _layer_groups(ctx, args, layer_idx, positions, model_inputs):
            raw["groups"] += 1
            state = group["state"]
            gain_pre, gain_post = state.single_block_gains()
            raw["local_pre"].append(group["local_pre"].detach().cpu())
            raw["local_post"].append(group["local_post"].detach().cpu())
            raw["gain_pre"].append(gain_pre.detach().cpu())
            raw["gain_post"].append(gain_post.detach().cpu())
            ks = {
                str(frac): max(1, min(state.num_blocks, int(round(frac * state.num_blocks))))
                for frac in args.fractions
            }

            if args.greedy_oracle:
                for objective in ("pre", "post"):
                    if objective == "post" and args.no_post_wo:
                        continue
                    curve = state.greedy_curve(ks.values(), objective=objective)
                    for frac, k in ks.items():
                        raw["greedy_oracle"][objective][frac].append(curve[k])

            for delta_kind in args.delta_kinds:
                scores, diagnostics = compute_scores(
                    group["p_hat"],
                    group[f"delta_{delta_kind}"],
                    group["b_c"],
                    group["b_all"],
                    group["sizes"],
                )
                raw["diagnostics"][delta_kind].append(diagnostics)
                for method, score in scores.items():
                    raw["scores"][delta_kind][method].append(score.detach().cpu())
                    raw["rank_stats"][delta_kind][method].append(
                        {
                            "local_pre": spearman(score, group["local_pre"]),
                            "local_post": spearman(score, group["local_post"]),
                            "gain_pre": spearman(score, gain_pre),
                            "gain_post": spearman(score, gain_post),
                        }
                    )

                for frac, k in ks.items():
                    full_mask = topk_mask(scores["full"], k)
                    term1_mask = topk_mask(scores["term1"], k)
                    term2_mask = topk_mask(scores["term2"], k)
                    for method, score in scores.items():
                        selected = topk_mask(score, k)
                        pre_error, post_error = state.error(selected)
                        local_pre_metrics = topk_metrics(score, group["local_pre"], k)
                        local_post_metrics = topk_metrics(score, group["local_post"], k)
                        gain_pre_metrics = topk_metrics(score, gain_pre, k)
                        gain_post_metrics = topk_metrics(score, gain_post, k)
                        raw["records"][delta_kind][method][frac].append(
                            {
                                "pre_error": pre_error,
                                "post_error": post_error,
                                **{f"local_pre_{key}": value for key, value in local_pre_metrics.items()},
                                **{f"local_post_{key}": value for key, value in local_post_metrics.items()},
                                **{f"gain_pre_{key}": value for key, value in gain_pre_metrics.items()},
                                **{f"gain_post_{key}": value for key, value in gain_post_metrics.items()},
                                "agreement_full": float((selected & full_mask).sum()) / k,
                                "agreement_term1": float((selected & term1_mask).sum()) / k,
                                "agreement_term2": float((selected & term2_mask).sum()) / k,
                            }
                        )

        result["layers"][str(layer_idx)] = _summarize_layer(
            raw, args.fractions, args.delta_kinds
        )
        print(f"[term-ablation] layer={layer_idx} done: {raw['groups']} groups", flush=True)

    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    markdown_path.write_text(_markdown(result, args))
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")
    return result


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
