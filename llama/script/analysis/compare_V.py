"""
Compare baseline H2O routing and optimal per-cluster scalar V routing upper bounds
under V-similarity clustering.

When a token is evicted, it is merged into the kept heavy-hitter token whose V is
most similar (largest dot product). Routing refinement still follows compare_count_optV:
for each cluster i at each (head, pos), solve
    r_i* = <tilde_V_i, g_i> / ||tilde_V_i||^2
where
    g_i = sum_{j in C_i} alpha_full_j * V_j.

Supported tilde_V choices:
- hh   : heavy-hitter token V
- avg  : mean V over cluster members
- wavg : alpha_full-weighted cluster mean V
"""

import argparse
import os

import torch

from .attention import (
    build_qk_routing_alpha,
    gen_mask_h2o_with_belong,
)
from .compare_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    plot_per_pos_two_lines,
    resolve_head_indices,
    resolve_output_dir,
    save_per_pos_metric_tsv,
    validate_common_args,
)
from .compare_count_optV import (
    build_hh_mask,
    build_tilde_v,
    canonicalize_belong,
    compute_cluster_gt,
    compute_full_attention_alpha,
    compute_recent_contrib,
    optimal_scalar_output,
    r_star_stats,
    extract_hh_positions,
    v_l2_per_pos_from_v,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimal per-cluster scalar V upper bounds under V clustering."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    parser.add_argument(
        "--tilde-v",
        nargs="+",
        choices=["hh", "avg", "wavg"],
        default=["hh", "avg", "wavg"],
        help="tilde_V choices to evaluate.",
    )
    parser.add_argument(
        "--save-r-star",
        action="store_true",
        default=False,
        help="Save per-cluster optimal scalar tensors.",
    )
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_common_args(
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
    output_dir = resolve_output_dir(
        args=args,
        head_idx=head_idx,
        compare_tag="compare_V",
        include_loss_type=True,
    )

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
            build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask_h2o_with_belong(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                head_idx=hi,
                budget=args.budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
                merge_metric="v",
            )[0],
        )

    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong, count, _hh_sumv_idx, _hh_sumv_val = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        merge_metric="v",
        return_hh_sumv=True,
        return_hh_sumk=False,
    )

    alpha_base = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        mask=route_mask,
        device=ctx.device,
    )

    alpha_full = compute_full_attention_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    visible = int(args.seq_len * args.budget)
    if args.adaptive_budget and (args.layer == 0 or args.layer == 1):
        visible = args.seq_len
    recent_budget = visible // 2

    hh_positions = extract_hh_positions(route_mask, recent_budget, pos_list)
    hh_mask = build_hh_mask(hh_positions, ctx.device)

    belong_root = canonicalize_belong(belong, pos_list)

    g, cluster_w = compute_cluster_gt(
        alpha_full=alpha_full,
        v_head=v_head,
        belong=belong_root,
        hh_positions=hh_positions,
    )
    recent_gt = compute_recent_contrib(
        alpha_full=alpha_full,
        v_head=v_head,
        pos_list=pos_list,
        recent_budget=recent_budget,
    )

    v_base = alpha_base.float() @ v_head.float()
    base_metric = v_l2_per_pos_from_v(v_base, v_gt)

    results = {}
    for choice in args.tilde_v:
        tv = build_tilde_v(
            choice=choice,
            v_head=v_head,
            belong=belong_root,
            hh_positions=hh_positions,
            g=g,
            cluster_w=cluster_w,
        )

        out_cluster, r_star, residual = optimal_scalar_output(tv, g, hh_mask)
        out_approx = out_cluster + recent_gt
        metric = v_l2_per_pos_from_v(out_approx, v_gt)
        r_mean, r_std = r_star_stats(r_star, hh_mask)

        valid_residual = residual * hh_mask.float()
        n_hh_per_pos = hh_mask.float().sum(-1)
        mean_residual = (valid_residual.sum(-1) / n_hh_per_pos.clamp_min(1.0)).mean(0)

        results[choice] = {
            "metric": metric,
            "r_star": r_star,
            "mean_residual": mean_residual,
            "r_mean": r_mean,
            "r_std": r_std,
        }

    print("===== Compare-V Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(f"mean base v_l2={float(base_metric.mean().item()):.8e}")
    for choice in args.tilde_v:
        metric = results[choice]["metric"]
        print(
            f"mean Vcluster_UB_{choice} v_l2={float(metric.mean().item()):.8e}, "
            f"mean improvement={float((base_metric - metric).mean().item()):.8e}, "
            f"r_mean={results[choice]['r_mean']:.4f}, r_std={results[choice]['r_std']:.4f}"
        )

    if "wavg" in results:
        print(
            "wavg sanity (mean v_l2, should be ~0): "
            f"{float(results['wavg']['metric'].mean().item()):.8e}"
        )

    os.makedirs(output_dir, exist_ok=True)

    for choice in args.tilde_v:
        metric = results[choice]["metric"]

        per_pos_path = os.path.join(output_dir, f"per_pos_v_l2_{choice}.tsv")
        save_per_pos_metric_tsv(
            out_path=per_pos_path,
            pos_list=pos_list,
            base_metric=base_metric,
            other_metric=metric,
            other_name=f"Vcluster_UB_{choice}",
        )

        plot_path = os.path.join(output_dir, f"per_pos_v_l2_{choice}.png")
        plot_per_pos_two_lines(
            out_path=plot_path,
            pos_list=pos_list,
            y1=base_metric,
            y2=metric,
            label1="base_v_l2",
            label2=f"Vcluster_UB_{choice}_v_l2",
            title="Per-Position V-L2: Base vs V-cluster UB_{} with budget={:.2f}".format(
                choice, args.budget
            ),
            dpi=args.plot_dpi,
        )

        print(f"Saved per-pos v_l2 table to: {per_pos_path}")
        print(f"Saved per-pos v_l2 plot to: {plot_path}")

    belong_path = os.path.join(output_dir, "belong.pt")
    torch.save(belong.detach().cpu(), belong_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "metric_name": "v_l2",
        "merge_metric": "v",
        "mean_base_metric": float(base_metric.mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "belong": belong.detach().cpu(),
        "belong_root": belong_root.detach().cpu(),
        "hh_mask": hh_mask.detach().cpu(),
        "cluster_w": cluster_w.detach().cpu(),
        "count": count.detach().cpu(),
    }
    for choice in args.tilde_v:
        metric = results[choice]["metric"]
        stats[f"mean_vcluster_ub_{choice}_metric"] = float(metric.mean().item())
        stats[f"mean_improvement_{choice}"] = float((base_metric - metric).mean().item())
        stats[f"vcluster_ub_{choice}_metric_per_pos"] = metric.detach().cpu()
        stats[f"mean_res_{choice}"] = results[choice]["mean_residual"].detach().cpu()
        stats[f"r_mean_{choice}"] = results[choice]["r_mean"]
        stats[f"r_std_{choice}"] = results[choice]["r_std"]
        if args.save_r_star:
            stats[f"r_star_{choice}"] = results[choice]["r_star"].detach().cpu()

    stats_path = os.path.join(output_dir, "compare_V_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
