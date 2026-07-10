"""
Compare baseline H2O routing and oracle refined routing.

For heavy-hitter keys j at each (head, pos):
- avgK_j = sumK_j / C_j
- delta_k(x) = k_x - avgK_j, for x in cluster(j)
- Routing logits on HH representatives:
      q·avgK_j + logsumexp_x(q·delta_k(x))
Value vectors use avgV_j = sumV_j / C_j on HH representatives.
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
        description="Compare baseline H2O routing and oracle refined routing."
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


def resolve_roots_from_belong_row(belong_row, total_available):
    parents = belong_row[:total_available].to(torch.long)
    roots = torch.empty_like(parents)

    for x in range(total_available):
        path = []
        r = x
        while True:
            p = int(parents[r].item())
            if p < 0 or p == r:
                break
            path.append(r)
            r = p
        roots[x] = r
        for n in path:
            parents[n] = r

    return roots


def build_oracle_refined_alpha(
    qk_logits,
    q_head,
    k_head,
    mask,
    count,
    belong,
    hh_sumk_idx,
    hh_sumk_val,
    pos_list,
):
    n_heads, n_pos, _seq_len = qk_logits.shape
    if mask.shape != qk_logits.shape:
        raise ValueError(f"mask shape mismatch: got {tuple(mask.shape)} expected {tuple(qk_logits.shape)}")
    if count.shape != qk_logits.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}"
        )
    if belong.shape != qk_logits.shape:
        raise ValueError(
            f"belong shape mismatch: got {tuple(belong.shape)} expected {tuple(qk_logits.shape)}"
        )
    if q_head.shape[0] != n_heads or q_head.shape[1] != n_pos:
        raise ValueError(
            f"q_head shape mismatch: got {tuple(q_head.shape)} expected ({n_heads}, {n_pos}, d)"
        )
    if k_head.shape[0] != n_heads:
        raise ValueError(
            f"k_head shape mismatch: got {tuple(k_head.shape)} expected ({n_heads}, seq, d)"
        )

    logits = qk_logits.to(torch.float32).clone()
    mask_f = mask.to(torch.float32)
    scale = float(q_head.shape[-1]) ** 0.5

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            idx = hh_sumk_idx[h][i]
            sumk = hh_sumk_val[h][i]
            if idx is None or sumk is None or idx.numel() == 0:
                continue

            roots = resolve_roots_from_belong_row(belong[h, i], total_available)
            q = q_head[h, i].float()

            for t, rep_tensor in enumerate(idx):
                rep = int(rep_tensor.item())
                c = count[h, i, rep].to(torch.float32).clamp_min(1.0)
                avgk = sumk[t].float() / c

                member_mask = roots == rep
                member_idx = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
                if member_idx.numel() == 0:
                    continue

                delta_k = k_head[h, member_idx].float() - avgk.unsqueeze(0)
                corr = torch.logsumexp((delta_k @ q) / scale, dim=0)
                q_avgk = (q * avgk).sum() / scale
                logits[h, i, rep] = q_avgk + corr

    return F.softmax(logits + mask_f, dim=-1)


def build_avgv_refined_v(alpha, v_head, hh_sumv_idx, hh_sumv_val, count):
    n_heads, n_pos, seq_len = alpha.shape
    if v_head.shape[0] != n_heads or v_head.shape[1] != seq_len:
        raise ValueError(
            f"v_head shape mismatch: got {tuple(v_head.shape)} expected heads={n_heads}, seq={seq_len}, d=*"
        )
    if count.shape != alpha.shape:
        raise ValueError(
            f"count shape mismatch: got {tuple(count.shape)} expected {tuple(alpha.shape)}"
        )

    v_new = alpha.float() @ v_head.float()
    for h in range(n_heads):
        for i in range(n_pos):
            idx = hh_sumv_idx[h][i]
            val_sum = hh_sumv_val[h][i]
            if idx is None or val_sum is None or idx.numel() == 0:
                continue

            c = count[h, i, idx].to(torch.float32).clamp_min(1.0).unsqueeze(-1)
            val_avg = val_sum.float() / c
            w = alpha[h, i, idx].float().unsqueeze(-1)
            delta = val_avg - v_head[h, idx].float()
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
        compare_tag="compare_count_oracle",
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
    k_head = layer_ctx.rope_qkv[args.layer]["k"].to(ctx.device)[0][head_idx].float()

    alpha_oracle = build_oracle_refined_alpha(
        qk_logits=qk_logits,
        q_head=q_head,
        k_head=k_head,
        mask=route_mask,
        count=count,
        belong=belong,
        hh_sumk_idx=hh_sumv_idx,
        hh_sumk_val=hh_sumk_val,
        pos_list=pos_list,
    )

    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[args.layer]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    v_base = alpha_base.float() @ v_head.float()
    v_oracle = build_avgv_refined_v(
        alpha=alpha_oracle,
        v_head=v_head,
        hh_sumv_idx=hh_sumv_idx,
        hh_sumv_val=hh_sumv_val,
        count=count,
    )

    base_metric = v_l2_per_pos_from_v(v_base, v_gt)
    oracle_metric = v_l2_per_pos_from_v(v_oracle, v_gt)

    print("===== Compare-Count-Oracle Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean oracle v_l2={float(oracle_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - oracle_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=oracle_metric,
        other_name="oracle",
    )

    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=oracle_metric,
        label1="base_v_l2",
        label2="oracle_v_l2",
        title="Per-Position V-L2: Base vs Oracle with budget={:.2f}".format(args.budget),
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
        "mean_oracle_metric": float(oracle_metric.mean().item()),
        "mean_improvement": float((base_metric - oracle_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "oracle_metric_per_pos": oracle_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_count_oracle_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved per-pos v_l2 plot to: {plot_path}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
