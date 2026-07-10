"""
Compare baseline H2O routing and optimal per-cluster scalar V routing upper bounds.

For each cluster i at each (head, pos), solve:
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
from typing import List

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimal per-cluster scalar V upper bounds."
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


def compute_full_attention_alpha(ctx, layer_idx, head_idx, pos_list, device):
    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    qk_sel = qk_scores[head_idx][:, pos_list, :].to(torch.float32)

    n_pos, seq_len = len(pos_list), qk_sel.shape[-1]
    causal = torch.full((n_pos, seq_len), float("-inf"), device=device)
    for i, pos in enumerate(pos_list):
        causal[i, : pos + 1] = 0.0

    return F.softmax(qk_sel + causal.unsqueeze(0), dim=-1)


def extract_hh_positions(mask, recent_budget, pos_list):
    n_heads, _, _ = mask.shape
    hh_positions = []
    for h in range(n_heads):
        head_hh = []
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            recent_start = max(0, total_available - recent_budget)
            kept = ~torch.isneginf(mask[h, i, :total_available])
            kept[recent_start:total_available] = False
            head_hh.append(kept.nonzero(as_tuple=False).squeeze(-1).tolist())
        hh_positions.append(head_hh)
    return hh_positions


def build_hh_mask(hh_positions, device):
    n_heads = len(hh_positions)
    n_pos = len(hh_positions[0]) if n_heads > 0 else 0
    max_hh = max((len(hh_positions[h][i]) for h in range(n_heads) for i in range(n_pos)), default=0)

    hh_mask = torch.zeros(n_heads, n_pos, max_hh, dtype=torch.bool, device=device)
    for h in range(n_heads):
        for i in range(n_pos):
            n = len(hh_positions[h][i])
            if n > 0:
                hh_mask[h, i, :n] = True
    return hh_mask


def canonicalize_belong(belong, pos_list):
    root = belong.clone()
    n_heads, n_pos, _ = root.shape

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            row = root[h, i, :total_available]

            while True:
                parent = row[row]
                if torch.equal(parent, row):
                    break
                row = parent

            root[h, i, :total_available] = row

    return root


def compute_cluster_gt(alpha_full, v_head, belong, hh_positions):
    n_heads, n_pos, _ = alpha_full.shape
    d = v_head.shape[-1]
    max_hh = max((len(hh_positions[h][i]) for h in range(n_heads) for i in range(n_pos)), default=0)

    if max_hh == 0:
        z1 = torch.zeros(n_heads, n_pos, 0, d, device=alpha_full.device)
        z2 = torch.zeros(n_heads, n_pos, 0, device=alpha_full.device)
        return z1, z2

    g = torch.zeros(n_heads, n_pos, max_hh, d, device=alpha_full.device)
    cluster_w = torch.zeros(n_heads, n_pos, max_hh, device=alpha_full.device)

    for h in range(n_heads):
        for i in range(n_pos):
            hh_ids = hh_positions[h][i]
            for slot, hh in enumerate(hh_ids):
                members = (belong[h, i] == hh).nonzero(as_tuple=False).squeeze(-1)
                if members.numel() == 0:
                    continue
                w = alpha_full[h, i, members].float()
                vm = v_head[h, members].float()
                g[h, i, slot] = (w.unsqueeze(-1) * vm).sum(0)
                cluster_w[h, i, slot] = w.sum()

    return g, cluster_w


def build_tilde_v(choice, v_head, belong, hh_positions, g, cluster_w):
    n_heads, n_pos, n_hh, d = g.shape

    if choice == "wavg":
        return g / cluster_w.clamp_min(1e-9).unsqueeze(-1)

    tv = torch.zeros(n_heads, n_pos, n_hh, d, device=v_head.device)
    for h in range(n_heads):
        for i in range(n_pos):
            hh_ids = hh_positions[h][i]
            for slot, hh in enumerate(hh_ids):
                if choice == "hh":
                    tv[h, i, slot] = v_head[h, hh].float()
                elif choice == "avg":
                    members = (belong[h, i] == hh).nonzero(as_tuple=False).squeeze(-1)
                    if members.numel() == 0:
                        tv[h, i, slot] = v_head[h, hh].float()
                    else:
                        tv[h, i, slot] = v_head[h, members].float().mean(0)
                else:
                    raise ValueError(f"Unsupported tilde_v choice: {choice}")
    return tv


def compute_recent_contrib(alpha_full, v_head, pos_list, recent_budget):
    n_heads, n_pos, _ = alpha_full.shape
    d = v_head.shape[-1]
    out = torch.zeros(n_heads, n_pos, d, device=alpha_full.device)

    for i, pos in enumerate(pos_list):
        total_available = pos + 1
        recent_start = max(0, total_available - recent_budget)
        if recent_start >= total_available:
            continue
        idx = torch.arange(recent_start, total_available, device=alpha_full.device)
        w = alpha_full[:, i, idx].float()  # [H, R]
        v = v_head[:, idx, :].float()      # [H, R, d]
        out[:, i] = (w.unsqueeze(-1) * v).sum(1)

    return out


def optimal_scalar_output(tilde_v, g, hh_mask):
    tv_norm2 = (tilde_v * tilde_v).sum(-1).clamp_min(1e-12)
    dot = (tilde_v * g).sum(-1)
    r_star = (dot / tv_norm2) * hh_mask.float()

    out_approx = (r_star.unsqueeze(-1) * tilde_v).sum(-2)
    residual = torch.norm(r_star.unsqueeze(-1) * tilde_v - g, p=2, dim=-1)
    return out_approx, r_star, residual


def v_l2_per_pos_from_v(v_new, v_gt):
    return torch.norm(v_new - v_gt.float(), p=2, dim=-1).mean(dim=0)


def r_star_stats(r_star, hh_mask):
    vals = r_star[hh_mask]
    if vals.numel() == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


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
        compare_tag="compare_count_optV",
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

    route_mask, belong, count, hh_sumv_idx, hh_sumv_val = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
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

    print("===== Compare-Count-OptV Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(f"mean base v_l2={float(base_metric.mean().item()):.8e}")
    for choice in args.tilde_v:
        m = results[choice]["metric"]
        print(
            f"mean UB_{choice} v_l2={float(m.mean().item()):.8e}, "
            f"mean improvement={float((base_metric - m).mean().item()):.8e}, "
            f"r_mean={results[choice]['r_mean']:.4f}, r_std={results[choice]['r_std']:.4f}"
        )

    if "wavg" in results:
        print(f"wavg sanity (mean v_l2, should be ~0): {float(results['wavg']['metric'].mean().item()):.8e}")

    os.makedirs(output_dir, exist_ok=True)

    for choice in args.tilde_v:
        metric = results[choice]["metric"]

        per_pos_path = os.path.join(output_dir, f"per_pos_v_l2_{choice}.tsv")
        save_per_pos_metric_tsv(
            out_path=per_pos_path,
            pos_list=pos_list,
            base_metric=base_metric,
            other_metric=metric,
            other_name=f"UB_{choice}",
        )

        plot_path = os.path.join(output_dir, f"per_pos_v_l2_{choice}.png")
        plot_per_pos_two_lines(
            out_path=plot_path,
            pos_list=pos_list,
            y1=base_metric,
            y2=metric,
            label1="base_v_l2",
            label2=f"UB_{choice}_v_l2",
            title="Per-Position V-L2: Base vs UB_{} with budget={:.2f}".format(
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
        "mean_base_metric": float(base_metric.mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "belong": belong.detach().cpu(),
        "belong_root": belong_root.detach().cpu(),
        "hh_mask": hh_mask.detach().cpu(),
        "cluster_w": cluster_w.detach().cpu(),
    }
    for choice in args.tilde_v:
        metric = results[choice]["metric"]
        stats[f"mean_ub_{choice}_metric"] = float(metric.mean().item())
        stats[f"mean_improvement_{choice}"] = float((base_metric - metric).mean().item())
        stats[f"ub_{choice}_metric_per_pos"] = metric.detach().cpu()
        stats[f"mean_res_{choice}"] = results[choice]["mean_residual"].detach().cpu()
        stats[f"r_mean_{choice}"] = results[choice]["r_mean"]
        stats[f"r_std_{choice}"] = results[choice]["r_std"]
        if args.save_r_star:
            stats[f"r_star_{choice}"] = results[choice]["r_star"].detach().cpu()

    stats_path = os.path.join(output_dir, "compare_count_optV_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
