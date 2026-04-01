"""
Visualize and compare optimal routing vs baseline routing for a specific layer.
print per-head image.
- attnetion map
- sparisity stats
- overlap grid

prefix layer 0 ~ L-1 with optimal routing, then training on layer L.
can prefill with baseline routing
"""
import argparse
import os

import torch

from .attention import (
    build_qk_routing_alpha,
    gen_mask,
    get_attention_map_after_rope,
    optimize_alpha_star,
)
from .compare_overlap import (
    build_diff_v_map,
    compute_overlap_stats,
    plot_overlap_grid,
    print_signed_topk_report,
    summarize_diff_v_topk_per_head,
)
from .compare_routing import plot_routing_grid
from .compare_sparsity import (
    compute_per_pos_sparsity_curves,
    plot_sparsity_curves_grid,
    print_sparsity_report,
    summarize_sparsity_global,
    summarize_sparsity_per_head,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import get_result_path, load_context, normalize_budget_key
from .sanity import build_modified_attn_hidden, move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare optimal routing and baseline routing for one layer/head."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )
    parser.add_argument("--adaptive-budget", action="store_true")

    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Single head index to analyze. If omitted and --heads not set, run all heads.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="A list of head indices to analyze. Overrides --head.",
    )
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="optimal_saved",
        choices=["optimal_saved", "baseline_rebuild"],
        help=(
            "How to prepare patches before target layer. "
            "optimal_saved: load saved optimal patch_hidden for layers < target; "
            "baseline_rebuild: rebuild baseline patches online for layers < target."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k used for overlap. Default: visible=int(seq_len*budget)",
    )
    parser.add_argument(
        "--pos-start",
        type=int,
        default=0,
        help="Start position (inclusive) for analysis.",
    )
    parser.add_argument(
        "--pos-end",
        type=int,
        default=None,
        help="End position (exclusive) for analysis. Default: seq_len.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--alpha-viz",
        type=str,
        default="log",
        choices=["linear", "log", "row_log"],
        help=(
            "Visualization scale for baseline/optimal alpha heatmaps. "
            "linear: raw probs; log: log10(probs); row_log: log10(probs / row_max)."
        ),
    )
    parser.add_argument(
        "--diff-log-eps",
        type=float,
        default=1e-2,
        help="Scale in signed-log diff map: sign(x)*log10(1 + |x|/eps).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sparsity-thresholds",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4],
        help="Thresholds for density check: report fraction of weights above each threshold.",
    )
    parser.add_argument(
        "--sparsity-mass-levels",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Mass levels for top-k coverage ratio, e.g. 0.9 means minimal k covering 90% mass.",
    )
    return parser.parse_args()


def validate_args(args, num_layers, num_heads):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.head is not None and (args.head < 0 or args.head >= num_heads):
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.heads is not None:
        for h in args.heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Invalid --heads entry {h}; expected [0, {num_heads - 1}]")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        if args.heads is not None and len(args.heads) > 0:
            head_tag = f"heads_{len(set(args.heads))}"
        elif args.head is not None:
            head_tag = f"head{args.head}"
        else:
            head_tag = "heads_all"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/compare/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def resolve_head_indices(args, num_heads):
    if args.heads is not None and len(args.heads) > 0:
        return sorted(set(int(x) for x in args.heads))
    if args.head is not None:
        return [int(args.head)]
    return list(range(num_heads))


def load_saved_patch_hidden_for_layer(args, layer_idx, budget, device):
    path = get_result_path(
        layer_idx=layer_idx,
        dataset=args.dataset,
        start=args.start,
        adaptive_budget=args.adaptive_budget,
        strategy=args.strategy,
        loss_type=args.loss_type,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing optimal layer result for layer={layer_idx}. Expected file: {path}"
        )

    result = torch.load(path, map_location="cpu", weights_only=False)
    key = normalize_budget_key(result, budget)
    if key is None:
        raise KeyError(
            f"Budget {budget} not found in layer={layer_idx} result keys: {list(result.keys())}"
        )

    entry = result[key]
    if not isinstance(entry, dict) or "patch_hidden" not in entry:
        raise KeyError(
            f"layer={layer_idx}, budget={budget} has no patch_hidden in saved result entry."
        )
    return entry["patch_hidden"].to(device)


def build_optimal_saved_prefix_patches(args, target_layer, budget, device):
    patches = {}
    for layer_idx in range(target_layer):
        patches[layer_idx] = load_saved_patch_hidden_for_layer(args, layer_idx, budget, device)
    return patches


def build_baseline_prefix_patches(ctx, args, target_layer, pos_list, model_inputs):
    patches = {}
    head_idx = list(range(ctx.model_config.num_attention_heads))
    for layer_idx in range(target_layer):
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
            budget=args.budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )
        alpha_baseline = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=mask,
            device=ctx.device,
        )
        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_baseline,
            device=ctx.device,
        )
        print(f"[prefix baseline rebuild] layer {layer_idx} done")
    return patches


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )
    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)

    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty pos_list after applying --pos-start/--pos-end")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(args)

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
    else:
        prefix_patches = build_baseline_prefix_patches(
            ctx=ctx,
            args=args,
            target_layer=args.layer,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    mask = gen_mask(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        strategy=args.strategy,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
    )

    alpha_baseline = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        mask=mask,
        device=ctx.device,
    )

    alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        training_steps=args.training_steps,
        lr=args.lr,
        mask=mask,
        loss_type=args.loss_type,
        device=ctx.device,
    )

    qk_scores_all, qk_probs_all = get_attention_map_after_rope(
        ctx=layer_ctx,
        layer_idx=args.layer,
        causal=True,
        dtype=torch.float32,
        device=ctx.device,
    )
    qk_probs_selected = qk_probs_all[head_idx][:, pos_list, :]
    v_selected = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_abs = torch.norm(v_selected, p=2, dim=-1)

    visible = int(args.seq_len * args.budget)
    topk = visible if args.topk is None else args.topk
    if topk <= 0:
        raise ValueError(
            f"topk={topk} is invalid. Increase --budget or provide a positive --topk."
        )

    # summary, per_head_summaries, per_head = compute_overlap_stats(
    #     alpha_opt=alpha_opt,
    #     alpha_base=alpha_baseline,
    #     pos_list=pos_list,
    #     topk=topk,
    # )

    diff_v = build_diff_v_map(
        alpha_opt=alpha_opt,
        alpha_base=alpha_baseline,
        v_abs=v_abs,
    )
    # diff_v_top10_per_head = summarize_diff_v_topk_per_head(
    #     alpha_opt=alpha_opt,
    #     alpha_base=alpha_baseline,
    #     v_abs=v_abs,
    #     pos_list=pos_list,
    #     topk=10,
    # )

    # qk_raw_sparsity_per_head = summarize_sparsity_per_head(
    #     weights=qk_probs_selected,
    #     pos_list=pos_list,
    #     thresholds=args.sparsity_thresholds,
    #     mass_levels=args.sparsity_mass_levels,
    # )
    # qk_routing_sparsity_per_head = summarize_sparsity_per_head(
    #     weights=alpha_baseline,
    #     pos_list=pos_list,
    #     thresholds=args.sparsity_thresholds,
    #     mass_levels=args.sparsity_mass_levels,
    # )
    # optimal_sparsity_per_head = summarize_sparsity_per_head(
    #     weights=alpha_opt,
    #     pos_list=pos_list,
    #     thresholds=args.sparsity_thresholds,
    #     mass_levels=args.sparsity_mass_levels,
    # )

    # qk_raw_sparsity_global = summarize_sparsity_global(qk_raw_sparsity_per_head)
    # qk_routing_sparsity_global = summarize_sparsity_global(qk_routing_sparsity_per_head)
    # optimal_sparsity_global = summarize_sparsity_global(optimal_sparsity_per_head)

    # per_pos_mass_level = 0.9
    # qk_raw_per_pos_curves = compute_per_pos_sparsity_curves(
    #     weights=qk_probs_selected,
    #     pos_list=pos_list,
    #     mass_level=per_pos_mass_level,
    # )
    # qk_routing_per_pos_curves = compute_per_pos_sparsity_curves(
    #     weights=alpha_baseline,
    #     pos_list=pos_list,
    #     mass_level=per_pos_mass_level,
    # )
    # optimal_per_pos_curves = compute_per_pos_sparsity_curves(
    #     weights=alpha_opt,
    #     pos_list=pos_list,
    #     mass_level=per_pos_mass_level,
    # )

    mat_path = os.path.join(output_dir, "routing_grid.png")
    overlap_curve_path = os.path.join(output_dir, "topk_overlap_curve.png")
    sparsity_curve_path = os.path.join(output_dir, "sparsity_curves_per_pos.png")
    stats_path = os.path.join(output_dir, "overlap_stats.pt")

    plot_routing_grid(
        alpha_base=alpha_baseline,
        alpha_opt=alpha_opt,
        diff_v=diff_v,
        head_labels=head_idx,
        out_path=mat_path,
        dpi=args.dpi,
        alpha_viz=args.alpha_viz,
        diff_log_eps=args.diff_log_eps,
    )

    # plot_overlap_grid(
    #     per_head=per_head,
    #     head_labels=head_idx,
    #     out_path=overlap_curve_path,
    #     dpi=args.dpi,
    # )

    # plot_sparsity_curves_grid(
    #     head_labels=head_idx,
    #     pos_list=pos_list,
    #     qk_raw_curves=qk_raw_per_pos_curves,
    #     qk_routing_curves=qk_routing_per_pos_curves,
    #     optimal_curves=optimal_per_pos_curves,
    #     out_path=sparsity_curve_path,
    #     dpi=args.dpi,
    #     mass_level=per_pos_mass_level,
    # )

    # torch.save(
    #     {
    #         "summary": summary,
    #         "per_head_summary": per_head_summaries,
    #         "per_head": per_head,
    #         "layer": args.layer,
    #         "heads": head_idx,
    #         "budget": float(args.budget),
    #         "topk": int(topk),
    #         "prefix_mode": args.prefix_mode,
    #         "loss": losses,
    #         "diff_v_top10_per_head": diff_v_top10_per_head,
    #         "sparsity": {
    #             "thresholds": args.sparsity_thresholds,
    #             "mass_levels": args.sparsity_mass_levels,
    #             "per_pos_mass_level": per_pos_mass_level,
    #             "qk_raw_per_head": qk_raw_sparsity_per_head,
    #             "qk_routing_per_head": qk_routing_sparsity_per_head,
    #             "optimal_per_head": optimal_sparsity_per_head,
    #             "qk_raw_global": qk_raw_sparsity_global,
    #             "qk_routing_global": qk_routing_sparsity_global,
    #             "optimal_global": optimal_sparsity_global,
    #             "qk_raw_per_pos_curves": qk_raw_per_pos_curves,
    #             "qk_routing_per_pos_curves": qk_routing_per_pos_curves,
    #             "optimal_per_pos_curves": optimal_per_pos_curves,
    #         },
    #     },
    #     stats_path,
    # )

    base_row_sum_err = (alpha_baseline.sum(dim=-1) - 1.0).abs().max().item()
    opt_row_sum_err = (alpha_opt.sum(dim=-1) - 1.0).abs().max().item()

    print("===== Compare Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"topk={topk}, prefix_mode={args.prefix_mode}"
    )
    # print(
    #     f"mean overlap ratio={summary['mean_overlap_ratio']:.6f} +- {summary['std_overlap_ratio']:.6f}"
    # )
    # print(f"mean jaccard={summary['mean_jaccard']:.6f} +- {summary['std_jaccard']:.6f}")
    # print(f"num_heads={summary['num_heads']}")
    # print(
    #     "max row-sum error: "
    #     f"baseline_alpha={base_row_sum_err:.3e}, optimal_alpha={opt_row_sum_err:.3e}"
    # )
    # print(
    #     "global density_gt_1e-3: "
    #     f"qk_raw={qk_raw_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}, "
    #     f"qk_routing={qk_routing_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}, "
    #     f"optimal={optimal_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}"
    # )
    # print(
    #     f"global k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}: "
    #     f"qk_raw={qk_raw_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}, "
    #     f"qk_routing={qk_routing_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}, "
    #     f"optimal={optimal_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}"
    # )
    # print_sparsity_report(
    #     head_labels=head_idx,
    #     qk_raw_stats=qk_raw_sparsity_per_head,
    #     qk_routing_stats=qk_routing_sparsity_per_head,
    #     optimal_stats=optimal_sparsity_per_head,
    #     mass_levels=args.sparsity_mass_levels,
    # )
    # print_signed_topk_report(
    #     head_labels=head_idx,
    #     signed_topk=diff_v_top10_per_head,
    #     score_name="(optimal routing - attention routing) * |V|",
    # )
    # print(f"saved matrix plot: {mat_path}")
    # print(f"saved overlap curve: {overlap_curve_path}")
    # print(f"saved sparsity curve: {sparsity_curve_path}")
    # print(f"saved stats: {stats_path}")


if __name__ == "__main__":
    main()