"""
Compare baseline QK routing and directly optimized routing.

This script mirrors compare_q_linear.py, but instead of fitting a qWk
parameterization, it directly optimizes routing weights alpha via the
existing optimize_alpha_star(...) implementation in attention.py.
"""

import argparse
import os

import torch

from .attention import build_qk_routing_alpha, gen_mask, optimize_alpha_star
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
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import build_modified_attn_hidden, move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline QK routing and direct optimal routing."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args()


def v_l2_per_pos(alpha, v_head, v_gt):
    v_new = alpha.float() @ v_head.float()
    l2 = torch.norm(v_new - v_gt.float(), p=2, dim=-1)
    return l2.mean(dim=0)


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
        compare_tag="compare_optimal_routing",
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
            build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                head_idx=hi,
                strategy=args.strategy,
                budget=args.budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
            ),
        )

    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask = gen_mask(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        strategy=args.strategy,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
    )

    alpha_base = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        mask=route_mask,
        device=ctx.device,
    )

    alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        training_steps=args.training_steps,
        lr=args.lr,
        mask=route_mask,
        loss_type=args.loss_type,
        device=ctx.device,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    base_metric = v_l2_per_pos(alpha_base.detach().to(torch.float32), v_head, v_gt)
    optimal_metric = v_l2_per_pos(alpha_opt.detach().to(torch.float32), v_head, v_gt)

    patch_hidden = build_modified_attn_hidden(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        alpha=alpha_opt,
        device=ctx.device,
    )

    print("===== Compare-Optimal-Routing Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, loss_type={args.loss_type}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean optimal v_l2={float(optimal_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - optimal_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=optimal_metric,
        other_name="optimal",
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=optimal_metric,
        label1="base_v_l2",
        label2="optimal_v_l2",
        title=f"Per-Position V-L2: Base vs Optimal with budget={args.budget:g}",
        dpi=args.plot_dpi,
    )

    alpha_path = os.path.join(output_dir, "alpha_optimal.pt")
    torch.save(alpha_opt.detach().cpu(), alpha_path)

    patch_hidden_path = os.path.join(output_dir, "patch_hidden_optimal.pt")
    torch.save(patch_hidden.detach().cpu(), patch_hidden_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "loss_type": args.loss_type,
        "metric_name": "v_l2",
        "alpha_optimal": alpha_opt.detach().cpu(),
        "losses": losses,
        "mean_base_metric": float(base_metric.mean().item()),
        "mean_optimal_metric": float(optimal_metric.mean().item()),
        "mean_improvement": float((base_metric - optimal_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "optimal_metric_per_pos": optimal_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_optimal_routing_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved per-pos v_l2 figure to: {plot_path}")
    print(f"Saved optimal alpha to: {alpha_path}")
    print(f"Saved optimal patch hidden to: {patch_hidden_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
