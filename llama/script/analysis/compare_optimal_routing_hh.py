"""
Compare baseline QK routing and HH-only optimal routing.

Only heavy-hitter (HH) routing is trainable. Recent-token routing is kept
exactly equal to the baseline routing and is not optimized.

This script is intended for H2O routing, where visible keys are split into:
- recent window: fixed baseline alpha
- heavy hitters: trainable alpha over the remaining probability mass
"""

import argparse
import os

import torch
from torch.nn import functional as F

from .attention import build_qk_routing_alpha, gen_mask
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
        description="Compare baseline QK routing and HH-only optimal routing."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
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


def compute_recent_budget(seq_len, budget, adaptive_budget, layer_idx):
    visible = int(seq_len * budget)
    if adaptive_budget and (layer_idx == 0 or layer_idx == 1):
        visible = seq_len
    return visible // 2


def build_hh_recent_masks(route_mask, pos_list, recent_budget):
    n_heads, n_pos, seq_len = route_mask.shape
    hh_mask = torch.zeros(n_heads, n_pos, seq_len, dtype=torch.bool, device=route_mask.device)
    recent_mask = torch.zeros(n_heads, n_pos, seq_len, dtype=torch.bool, device=route_mask.device)

    visible_mask = ~torch.isneginf(route_mask)
    for i, pos in enumerate(pos_list):
        total_available = pos + 1
        recent_start = max(0, total_available - recent_budget)
        if recent_start < total_available:
            recent_mask[:, i, recent_start:total_available] = visible_mask[:, i, recent_start:total_available]
        if recent_start > 0:
            hh_mask[:, i, :recent_start] = visible_mask[:, i, :recent_start]

    return hh_mask, recent_mask


def build_hh_only_alpha(base_alpha, hh_alpha, hh_mask, recent_mask):
    recent_alpha = base_alpha * recent_mask.float()
    recent_mass = recent_alpha.sum(dim=-1, keepdim=True)
    hh_mass = (1.0 - recent_mass).clamp_min(0.0)
    return recent_alpha + hh_alpha * hh_mass * hh_mask.float()


def optimize_alpha_hh_only(
    ctx,
    layer_idx,
    head_idx,
    pos_list,
    base_alpha,
    hh_mask,
    recent_mask,
    training_steps,
    lr,
    loss_type="v_l2",
    device=None,
):
    if device is None:
        device = ctx.device
    if isinstance(head_idx, int):
        head_idx = [head_idx]

    n_heads = len(head_idx)
    n_pos = len(pos_list)
    seq_len = base_alpha.shape[-1]
    hh_exists = hh_mask.any(dim=-1)

    hh_logits = torch.nn.Parameter(
        torch.zeros(n_heads, n_pos, seq_len, dtype=torch.float32, device=device)
    )
    opt = torch.optim.Adam([hh_logits], lr=lr)

    residual_attn_in = ctx.layer_input[layer_idx][0, pos_list].to(device)
    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)
    v_head = ctx.rope_qkv[layer_idx]["v"].to(device)[0][head_idx]
    layer = ctx.model.model.layers[layer_idx]
    gt_v = original[head_idx].detach().float()

    p_teacher = None
    logp_teacher = None
    p_teacher_v = None
    logp_teacher_v = None

    if loss_type == "logits_kl":
        with torch.no_grad():
            output = original.clone()
            hidden = layer.self_attn.o_proj(output.permute(1, 0, 2).reshape(len(pos_list), -1))
            hidden = hidden + residual_attn_in
            gt_logits = ctx.model.lm_head(hidden)
            p_teacher = F.softmax(gt_logits.float(), dim=-1).detach()
            logp_teacher = F.log_softmax(gt_logits.float(), dim=-1).detach()
    elif loss_type == "v_kl":
        with torch.no_grad():
            p_teacher_v = F.softmax(gt_v, dim=-1).detach()
            logp_teacher_v = F.log_softmax(gt_v, dim=-1).detach()
    elif loss_type == "v_l2":
        pass
    else:
        raise ValueError(f"Unknown loss_type {loss_type}. Supported: logits_kl, v_l2, v_kl")

    losses = []
    p_alpha = None
    base_alpha = base_alpha.to(device=device, dtype=torch.float32)
    recent_mask = recent_mask.to(device=device)

    for step in range(training_steps):
        large_neg = torch.full_like(hh_logits, -1e9)
        masked_logits = torch.where(hh_mask, hh_logits, large_neg)
        hh_alpha = F.softmax(masked_logits, dim=-1)
        alpha = build_hh_only_alpha(base_alpha, hh_alpha, hh_mask, recent_mask)

        no_hh_rows = ~hh_exists
        if torch.any(no_hh_rows):
            alpha = alpha.clone()
            alpha[no_hh_rows] = base_alpha[no_hh_rows]

        v_new = alpha @ v_head.float()
        v_new = v_new.to(original.dtype)

        if loss_type == "logits_kl":
            output = original.clone()
            output[head_idx] = v_new.to(v_head.dtype)
            hidden = output.permute(1, 0, 2).reshape(len(pos_list), -1)
            hidden = layer.self_attn.o_proj(hidden)
            hidden = hidden + residual_attn_in
            logits = ctx.model.lm_head(hidden)

            p_alpha = F.softmax(logits.float(), dim=-1)
            logp_student = F.log_softmax(logits.float(), dim=-1)
            loss = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1).mean()
        elif loss_type == "v_l2":
            loss = torch.norm(v_new.float() - gt_v, p=2, dim=-1).mean()
        else:
            logp_student_v = F.log_softmax(v_new.float(), dim=-1)
            loss = (p_teacher_v * (logp_teacher_v - logp_student_v)).sum(dim=-1).mean()

        if step % 100 == 0 or step == training_steps - 1:
            loss_v = float(loss.detach().float().cpu().item())
            losses.append((int(step), loss_v))
            print(f"[hh-opt] step={step:4d} loss={loss_v:.8f}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        large_neg = torch.full_like(hh_logits, -1e9)
        masked_logits = torch.where(hh_mask, hh_logits, large_neg)
        hh_alpha = F.softmax(masked_logits, dim=-1)
        alpha = build_hh_only_alpha(base_alpha, hh_alpha, hh_mask, recent_mask)
        if torch.any(~hh_exists):
            alpha[~hh_exists] = base_alpha[~hh_exists]

    return alpha, p_alpha, p_teacher, losses


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
        compare_tag="compare_optimal_routing_hh",
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

    recent_budget = compute_recent_budget(
        seq_len=args.seq_len,
        budget=args.budget,
        adaptive_budget=args.adaptive_budget,
        layer_idx=args.layer,
    )
    hh_mask, recent_mask = build_hh_recent_masks(route_mask, pos_list, recent_budget)

    alpha_opt_hh, _p_alpha, _p_teacher, losses = optimize_alpha_hh_only(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        base_alpha=alpha_base,
        hh_mask=hh_mask,
        recent_mask=recent_mask,
        training_steps=args.training_steps,
        lr=args.lr,
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
    optimal_metric = v_l2_per_pos(alpha_opt_hh.detach().to(torch.float32), v_head, v_gt)

    recent_diff = (alpha_opt_hh - alpha_base).abs() * recent_mask.float()
    max_recent_delta = float(recent_diff.max().item()) if recent_diff.numel() > 0 else 0.0

    patch_hidden = build_modified_attn_hidden(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        alpha=alpha_opt_hh,
        device=ctx.device,
    )

    hh_rows = hh_mask.any(dim=-1).float()
    avg_hh_count = float(hh_mask.float().sum(dim=-1)[hh_rows.bool()].mean().item()) if torch.any(hh_rows.bool()) else 0.0

    print("===== Compare-Optimal-Routing-HH Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, loss_type={args.loss_type}, recent_budget={recent_budget}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean optimal_hh v_l2={float(optimal_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - optimal_metric).mean().item()):.8e}, "
        f"avg_hh_count={avg_hh_count:.2f}, max_recent_delta={max_recent_delta:.3e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=optimal_metric,
        other_name="optimal_hh",
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=optimal_metric,
        label1="base_v_l2",
        label2="optimal_hh_v_l2",
        title=f"Per-Position V-L2: Base vs Optimal-HH with budget={args.budget:g}",
        dpi=args.plot_dpi,
    )

    alpha_path = os.path.join(output_dir, "alpha_optimal_hh.pt")
    torch.save(alpha_opt_hh.detach().cpu(), alpha_path)

    patch_hidden_path = os.path.join(output_dir, "patch_hidden_optimal_hh.pt")
    torch.save(patch_hidden.detach().cpu(), patch_hidden_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "loss_type": args.loss_type,
        "metric_name": "v_l2",
        "recent_budget": int(recent_budget),
        "max_recent_delta": max_recent_delta,
        "hh_mask": hh_mask.detach().cpu(),
        "recent_mask": recent_mask.detach().cpu(),
        "alpha_base": alpha_base.detach().cpu(),
        "alpha_optimal_hh": alpha_opt_hh.detach().cpu(),
        "losses": losses,
        "mean_base_metric": float(base_metric.mean().item()),
        "mean_optimal_metric": float(optimal_metric.mean().item()),
        "mean_improvement": float((base_metric - optimal_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "optimal_metric_per_pos": optimal_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_optimal_routing_hh_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved per-pos v_l2 figure to: {plot_path}")
    print(f"Saved HH-only optimal alpha to: {alpha_path}")
    print(f"Saved HH-only optimal patch hidden to: {patch_hidden_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
