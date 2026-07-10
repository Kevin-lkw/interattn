"""
Compare baseline H2O routing and count-refined H2O routing.

Count refinement rule (heavy hitter only):
    refined_logit = qk + log(C)
where C is the size of the belong-set mapped to that heavy hitter key.
Recent tokens are not adjusted.
"""

import argparse
import os

import torch
from torch.nn import functional as F

from ..attention import (
    build_qk_routing_alpha,
    gen_mask_h2o_with_belong,
    get_attention_map_after_rope,
)
from ..experiment_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
    resolve_output_dir,
    save_per_pos_metric_tsv,
    validate_common_args,
    plot_per_pos_two_lines,
)
from ..config import set_seed, str_to_torch_dtype
from ..online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ..runtime import load_context
from ..sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline H2O routing and count-refined H2O routing."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    return parser.parse_args()


def get_qk_logits(ctx, layer_idx, head_idx, pos_list, device):
    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    return qk_scores[head_idx][:, pos_list, :].to(torch.float32)


def build_count_refined_alpha(qk_logits, mask, belong, count, pos_list, recent_budget):
    n_heads, n_pos, seq_len = qk_logits.shape
    if mask.shape != qk_logits.shape:
        raise ValueError(f"mask shape mismatch: got {tuple(mask.shape)} expected {tuple(qk_logits.shape)}")
    if belong.shape != qk_logits.shape:
        raise ValueError(
            f"belong shape mismatch: got {tuple(belong.shape)} expected {tuple(qk_logits.shape)}"
        )
    if count.shape != qk_logits.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}"
        )

    logits = qk_logits.to(torch.float32).clone()
    mask_f = mask.to(torch.float32)

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            # import ipdb; ipdb.set_trace()
            total_available = pos + 1
            recent_start = max(0, total_available - recent_budget)

            row_belong = belong[h, i, :total_available]
            if (row_belong < 0).any():
                raise ValueError("belong contains invalid negative index in lower-triangular region")

            counts = count[h, i, :total_available]
            assert counts.sum() == total_available, "count should sum up to total_available"
            
            visible = ~torch.isneginf(mask_f[h, i, :total_available])
            hh_visible = visible.clone()
            hh_visible[recent_start:total_available] = False
            hh_idx = torch.nonzero(hh_visible, as_tuple=False).squeeze(-1)

            if len(hh_idx) > 0:
                c = counts[hh_idx]
                assert c.min() >= 1, "count should be >= 1 for visible tokens"
                logits[h, i, hh_idx] = logits[h, i, hh_idx] + torch.log(c.float())

    return F.softmax(logits + mask_f, dim=-1)


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
        compare_tag="compare_count",
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

    route_mask, belong, count = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
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

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    visible = int(args.seq_len * args.budget)
    if args.adaptive_budget and (args.layer == 0 or args.layer == 1):
        visible = args.seq_len
    recent_budget = visible // 2

    alpha_count = build_count_refined_alpha(
        qk_logits=qk_logits,
        mask=route_mask,
        belong=belong,
        pos_list=pos_list,
        recent_budget=recent_budget,
        count=count,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    base_metric = v_l2_per_pos(alpha_base.detach().to(torch.float32), v_head, v_gt)
    count_metric = v_l2_per_pos(alpha_count.detach().to(torch.float32), v_head, v_gt)

    print("===== Compare-Count Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean count v_l2={float(count_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - count_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=count_metric,
        other_name="count",
    )
    
    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=count_metric,
        label1="base_v_l2",
        label2="count_v_l2",
        title="Per-Position V-L2: Base vs Count-Refined",
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
        "mean_count_metric": float(count_metric.mean().item()),
        "mean_improvement": float((base_metric - count_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "count_metric_per_pos": count_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_count_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
