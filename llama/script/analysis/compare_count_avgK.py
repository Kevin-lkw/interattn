"""
Compare baseline H2O routing and avgK refined routing.

For heavy-hitter keys j at each (head, pos):
- Routing logits: q·avgK_j + log(C_j)
- Value vectors remain original heavy-hitter V (no avgV replacement)
where C_j is the cluster size (belong count).
"""

import argparse
import os

import torch
from torch.nn import functional as F

from .attention import (
    build_qk_routing_alpha,
    gen_mask_h2o_with_belong,
    get_attention_map_after_rope,
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
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline H2O routing and avgK refined routing."
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


def build_avgk_count_refined_alpha(
    qk_logits,
    q_head,
    mask,
    count,
    hh_sumk_idx,
    hh_sumk_val,
    pos_list,
    recent_budget,
):
    n_heads, n_pos, _seq_len = qk_logits.shape
    if mask.shape != qk_logits.shape:
        raise ValueError(f"mask shape mismatch: got {tuple(mask.shape)} expected {tuple(qk_logits.shape)}")
    if count.shape != qk_logits.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}"
        )
    if q_head.shape[0] != n_heads or q_head.shape[1] != n_pos:
        raise ValueError(
            f"q_head shape mismatch: got {tuple(q_head.shape)} expected ({n_heads}, {n_pos}, d)"
        )

    logits = qk_logits.to(torch.float32).clone()
    mask_f = mask.to(torch.float32)
    scale = float(q_head.shape[-1]) ** 0.5

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            recent_start = max(0, total_available - recent_budget)

            idx = hh_sumk_idx[h][i]
            sumk = hh_sumk_val[h][i]
            if idx is None or sumk is None or idx.numel() == 0:
                continue

            # Safety gate: ensure idx still corresponds to visible heavy-hitter keys.
            visible = ~torch.isneginf(mask_f[h, i, :total_available])
            hh_visible = visible.clone()
            hh_visible[recent_start:total_available] = False
            if hh_visible[idx].any().item() is False:
                continue

            c = count[h, i, idx].to(torch.float32).clamp_min(1.0)
            avgk = sumk.float() / c.unsqueeze(-1)
            q = q_head[h, i].float().unsqueeze(0)  # [1, d]
            q_avgk = (q * avgk).sum(dim=-1) / scale
            logits[h, i, idx] = q_avgk + torch.log(c)

    return F.softmax(logits + mask_f, dim=-1)


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
        compare_tag="compare_count_avgK",
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

    route_mask, belong, count, hh_sumv_idx, hh_sumv_val, hh_sumk_val = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        return_hh_sumv=True,
        return_hh_sumk=True,
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

    q_head = layer_ctx.rope_qkv[args.layer]["q"].to(ctx.device)[0][head_idx][:, pos_list, :].float()

    visible = int(args.seq_len * args.budget)
    if args.adaptive_budget and (args.layer == 0 or args.layer == 1):
        visible = args.seq_len
    recent_budget = visible // 2

    alpha_avgk = build_avgk_count_refined_alpha(
        qk_logits=qk_logits,
        q_head=q_head,
        mask=route_mask,
        count=count,
        hh_sumk_idx=hh_sumv_idx,
        hh_sumk_val=hh_sumk_val,
        pos_list=pos_list,
        recent_budget=recent_budget,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    v_base = alpha_base.float() @ v_head.float()
    v_avgk = alpha_avgk.float() @ v_head.float()

    base_metric = v_l2_per_pos_from_v(v_base, v_gt)
    avgk_metric = v_l2_per_pos_from_v(v_avgk, v_gt)

    print("===== Compare-Count-AvgK Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean avgK v_l2={float(avgk_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - avgk_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=avgk_metric,
        other_name="avgK",
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=avgk_metric,
        label1="base_v_l2",
        label2="avgK_v_l2",
        title="Per-Position V-L2: Base vs AvgK with budget={:.2f}".format(args.budget),
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
        "mean_avgk_metric": float(avgk_metric.mean().item()),
        "mean_improvement": float((base_metric - avgk_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "avgk_metric_per_pos": avgk_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_count_avgK_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved per-pos v_l2 plot to: {plot_path}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
