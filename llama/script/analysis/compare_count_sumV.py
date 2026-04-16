"""
Compare baseline H2O routing and sum-V refined H2O routing.

Sum-V refinement rule (heavy hitter only):
    V_hh(j) <- sum_{x: belong(x)=j} V(x)

Routing logits stay unchanged (baseline QK + mask). Since belong depends on query
position, each (head, pos) uses a different effective V table.
"""

import argparse
import os

import torch

from .attention import build_qk_routing_alpha, gen_mask_h2o_with_belong
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
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline H2O routing and sum-V refined H2O routing."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    return parser.parse_args()


def build_sumv_refined_v(alpha, v_head, hh_sumv_idx, hh_sumv_val):
    n_heads, n_pos, seq_len = alpha.shape
    if v_head.shape[0] != n_heads or v_head.shape[1] != seq_len:
        raise ValueError(
            f"v_head shape mismatch: got {tuple(v_head.shape)} expected heads={n_heads}, seq={seq_len}, d=*"
        )

    # Start from baseline alpha @ V, then only patch heavy-hitter keys with precomputed summed-V.
    v_new = alpha.float() @ v_head.float()  # [h, n_pos, hd]
    for h in range(n_heads):
        for i in range(n_pos):
            idx = hh_sumv_idx[h][i]
            val = hh_sumv_val[h][i]
            if idx is None or val is None or idx.numel() == 0:
                continue
            w = alpha[h, i, idx].float().unsqueeze(-1)  # [k, 1]
            delta = val.float() - v_head[h, idx].float()  # [k, hd]
            v_new[h, i] = v_new[h, i] + (w * delta).sum(dim=0)

    return v_new


def v_l2_per_pos_from_v(v_new, v_gt):
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
        compare_tag="compare_count_sumV",
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

    route_mask, belong, _count, hh_sumv_idx, hh_sumv_val = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        return_hh_sumv=True,
    )

    alpha_base = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        mask=route_mask,
        device=ctx.device,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    v_base = alpha_base.float() @ v_head.float()
    v_sumv = build_sumv_refined_v(
        alpha=alpha_base,
        v_head=v_head,
        hh_sumv_idx=hh_sumv_idx,
        hh_sumv_val=hh_sumv_val,
    )

    base_metric = v_l2_per_pos_from_v(v_base, v_gt)
    sumv_metric = v_l2_per_pos_from_v(v_sumv, v_gt)

    print("===== Compare-Count-SumV Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean sumV v_l2={float(sumv_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - sumv_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=sumv_metric,
        other_name="sumV",
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=sumv_metric,
        label1="base_v_l2",
        label2="sumV_v_l2",
        title="Per-Position V-L2: Base vs SumV",
        dpi=args.plot_dpi,
    )

    belong_path = os.path.join(output_dir, "belong.pt")
    torch.save(belong.detach().cpu(), belong_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "metric_name": "v_l2",
        "mean_base_metric": float(base_metric.mean().item()),
        "mean_sumv_metric": float(sumv_metric.mean().item()),
        "mean_improvement": float((base_metric - sumv_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "sumv_metric_per_pos": sumv_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_count_sumV_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved per-pos v_l2 plot to: {plot_path}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
