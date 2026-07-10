"""
Compare baseline kept-token routing and original full-attention routing restricted to kept tokens.

Assume the kept tokens are V1...Vm from the routing mask. Routing weights use the
full-attention softmax over all tokens, but we zero out dropped tokens and DO NOT
renormalize (so weights need not sum to 1).
"""

import argparse
import os

import torch

from ..attention import build_qk_routing_alpha, gen_mask
from .compare_count_optV import compute_full_attention_alpha, v_l2_per_pos_from_v
from ..experiment_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    plot_per_pos_two_lines,
    resolve_head_indices,
    resolve_output_dir,
    save_per_pos_metric_tsv,
    validate_common_args,
)
from ..config import set_seed, str_to_torch_dtype
from ..online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ..runtime import load_context
from ..sanity import move_model_inputs_to_device


def save_selected_attention_ratio_tsv(out_path, pos_list, head_idx, selected_attention_ratio):
    ratio = selected_attention_ratio.detach().float().cpu()
    with open(out_path, "w", encoding="utf-8") as f:
        head_cols = "\t".join([f"head_{h}" for h in head_idx])
        f.write(f"pos\tmean_selected_attention_ratio\t{head_cols}\n")
        for i, pos in enumerate(pos_list):
            vals = ratio[:, i]
            val_str = "\t".join([f"{float(x.item()):.8e}" for x in vals])
            f.write(f"{pos}\t{float(vals.mean().item()):.8e}\t{val_str}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline routing vs original full-attention routing over kept tokens."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o", "attention_topk"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
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
        compare_tag="compare_original",
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

    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
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

    alpha_full = compute_full_attention_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    keep_mask = (~torch.isneginf(route_mask)).float()
    alpha_original = alpha_full * keep_mask
    selected_attention_ratio = alpha_original.sum(dim=-1)
    selected_attention_ratio_per_pos = selected_attention_ratio.mean(dim=0)

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    v_base = alpha_base.float() @ v_head.float()
    base_metric = v_l2_per_pos_from_v(v_base, v_gt)

    v_original = alpha_original.float() @ v_head.float()
    original_metric = v_l2_per_pos_from_v(v_original, v_gt)

    print("===== Compare-Original Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(f"mean base v_l2={float(base_metric.mean().item()):.8e}")
    print(f"mean original v_l2={float(original_metric.mean().item()):.8e}")
    print(
        f"mean improvement={float((base_metric - original_metric).mean().item()):.8e}"
    )
    print(
        "selected token attention ratio: "
        f"mean={float(selected_attention_ratio.mean().item()):.8e}, "
        f"std={float(selected_attention_ratio.std(unbiased=False).item()):.8e}, "
        f"min={float(selected_attention_ratio.min().item()):.8e}, "
        f"max={float(selected_attention_ratio.max().item()):.8e}"
    )

    os.makedirs(output_dir, exist_ok=True)

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2_original.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=original_metric,
        other_name="original",
    )

    selected_ratio_path = os.path.join(output_dir, "selected_attention_ratio.tsv")
    save_selected_attention_ratio_tsv(
        out_path=selected_ratio_path,
        pos_list=pos_list,
        head_idx=head_idx,
        selected_attention_ratio=selected_attention_ratio,
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2_original.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=original_metric,
        label1="base_v_l2",
        label2="original_v_l2",
        title="Per-Position V-L2: Base vs Original (no renorm) with budget={:.2f}".format(
            args.budget
        ),
        dpi=args.plot_dpi,
    )

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved selected attention ratio table to: {selected_ratio_path}")
    print(f"Saved per-pos v_l2 plot to: {plot_path}")

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "metric_name": "v_l2",
        "mean_base_metric": float(base_metric.mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "mean_original_metric": float(original_metric.mean().item()),
        "original_metric_per_pos": original_metric.detach().cpu(),
        "keep_mask": keep_mask.detach().cpu(),
        "selected_attention_ratio": selected_attention_ratio.detach().cpu(),
        "selected_attention_ratio_per_pos": selected_attention_ratio_per_pos.detach().cpu(),
        "mean_selected_attention_ratio": float(selected_attention_ratio.mean().item()),
    }

    stats_path = os.path.join(output_dir, "compare_original_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
