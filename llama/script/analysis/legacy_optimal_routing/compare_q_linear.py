"""
Approximate routing with per-head trainable linear map W in QK logits.

For each selected head h, logits are changed from q_h k_h^T to (q_h W_h) k_h^T.
This script only supports tau-target=v_l2_gt, i.e., directly minimizing
||V_linear - V_gt||_2 without computing optimal routing.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from ..attention import build_qk_routing_alpha, gen_mask
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
        description="Compare QK routing and Q-linear routing (qWk) with v_l2_gt objective."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )

    parser.add_argument("--w-steps", type=int, default=600)
    parser.add_argument("--w-lr", type=float, default=1e-3)
    parser.add_argument("--w-l2", type=float, default=0.0)
    parser.add_argument(
        "--w-structure",
        type=str,
        default="full",
        choices=["full", "diag"],
        help="Constraint for per-head W in qWk. full: unconstrained matrix; diag: diagonal-only.",
    )

    parser.add_argument(
        "--tau-target",
        type=str,
        default="v_l2_gt",
        choices=["v_l2_gt"],
        help="Fixed target for this script.",
    )

    return parser.parse_args()


def build_qk_linear_logits(layer_ctx, layer_idx, head_idx, pos_list, w, device):
    q_all = layer_ctx.rope_qkv[layer_idx]["q"].to(device)[0][head_idx].float()  # [h, seq, d]
    k_all = layer_ctx.rope_qkv[layer_idx]["k"].to(device)[0][head_idx].float()  # [h, seq, d]
    w = w.to(device=device, dtype=torch.float32)

    q = q_all[:, pos_list, :]  # [h, n_pos, d]
    k = k_all  # [h, seq, d]

    # [h, n_pos, d]
    q_w = torch.einsum("hpd,hde->hpe", q, w)
    d = q_w.shape[-1]
    logits = torch.einsum("hpe,hse->hps", q_w, k) / math.sqrt(float(d))
    return logits


def build_causal_additive_mask(pos_list, seq_len, device):
    n_pos = len(pos_list)
    key_idx = torch.arange(seq_len, device=device).view(1, seq_len)
    pos_tensor = torch.tensor(pos_list, device=device).view(n_pos, 1)
    invalid = key_idx > pos_tensor
    cm = torch.zeros(n_pos, seq_len, device=device, dtype=torch.float32)
    cm[invalid] = float("-inf")
    return cm.unsqueeze(0)


def build_q_linear_alpha(layer_ctx, layer_idx, head_idx, pos_list, w, route_mask, device):
    logits = build_qk_linear_logits(layer_ctx, layer_idx, head_idx, pos_list, w, device)
    causal_mask = build_causal_additive_mask(pos_list, route_mask.shape[-1], device)

    safe_route_mask = route_mask.to(torch.float32)
    safe_route_mask = torch.where(
        torch.isneginf(safe_route_mask),
        torch.full_like(safe_route_mask, -1e9),
        safe_route_mask,
    )
    safe_causal = torch.where(
        torch.isneginf(causal_mask),
        torch.full_like(causal_mask, -1e9),
        causal_mask,
    )
    return F.softmax(logits + safe_route_mask + safe_causal, dim=-1)


def _project_w_structure_(w_param, w_structure):
    if w_structure == "diag":
        with torch.no_grad():
            diag = torch.diagonal(w_param.data, dim1=-2, dim2=-1)
            w_param.data = torch.diag_embed(diag)
    elif w_structure != "full":
        raise ValueError(f"Unknown w_structure: {w_structure}")


def optimize_w_v_l2(
    layer_ctx,
    layer_idx,
    head_idx,
    pos_list,
    route_mask,
    w_steps,
    w_lr,
    w_l2,
    device,
    w_structure="full",
):
    v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(device)[0][head_idx].float()  # [h, seq, d]
    v_gt = (
        layer_ctx.attn_output[layer_idx]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(device)[head_idx]
        .float()
    )  # [h, n_pos, d]

    h = len(head_idx)
    d = v_head.shape[-1]
    eye = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0).expand(h, -1, -1)
    w_param = torch.nn.Parameter(eye.clone())
    _project_w_structure_(w_param, w_structure)
    opt = torch.optim.Adam([w_param], lr=w_lr)

    history = []
    for step in range(int(w_steps)):
        alpha = build_q_linear_alpha(
            layer_ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            w=w_param,
            route_mask=route_mask,
            device=device,
        )
        v_new = alpha @ v_head
        loss = torch.norm(v_new - v_gt, p=2, dim=-1).mean()
        if float(w_l2) > 0.0:
            loss = loss + float(w_l2) * (w_param - eye).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_param], max_norm=1.0)
        opt.step()
        _project_w_structure_(w_param, w_structure)

        if step % 20 == 0 or step == int(w_steps) - 1:
            lv = float(loss.detach().cpu().item())
            history.append((int(step), lv))
            print(f"[w-opt] step={step:4d} loss={lv:.8f}")

    with torch.no_grad():
        w_final = w_param.detach().cpu()
        alpha_final = build_q_linear_alpha(
            layer_ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            w=w_final,
            route_mask=route_mask,
            device=device,
        )

    return w_final, alpha_final, history, v_head, v_gt


def v_l2_per_pos(alpha, v_head, v_gt):
    v_new = alpha.float() @ v_head.float()
    l2 = torch.norm(v_new - v_gt.float(), p=2, dim=-1)
    return l2.mean(dim=0)


def save_w_norm_tsv(out_path, w, head_idx):
    h = w.shape[0]
    d = w.shape[-1]
    eye = torch.eye(d, dtype=w.dtype).unsqueeze(0).expand(h, -1, -1)
    delta = w - eye
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("head\tw_delta_fro\n")
        for i, h_id in enumerate(head_idx):
            v = float(torch.norm(delta[i], p="fro").item())
            f.write(f"{h_id}\t{v:.8e}\n")


def plot_w_norm(out_path, w, head_idx, dpi=180):
    h = w.shape[0]
    d = w.shape[-1]
    eye = torch.eye(d, dtype=w.dtype).unsqueeze(0).expand(h, -1, -1)
    delta = w - eye
    y = [float(torch.norm(delta[i], p="fro").item()) for i in range(h)]

    plt.figure(figsize=(9, 4.8))
    plt.scatter(head_idx, y, s=30)
    plt.xlabel("head")
    plt.ylabel("||W - I||_F")
    plt.title("Per-Head Linear Delta Norm")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


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
        compare_tag="compare_q_linear",
        include_loss_type=True,
    )

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
            loss_type_override="v_l2",
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

    w, alpha_linear, w_history, v_head, v_gt = optimize_w_v_l2(
        layer_ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        route_mask=route_mask,
        w_steps=args.w_steps,
        w_lr=args.w_lr,
        w_l2=args.w_l2,
        device=ctx.device,
        w_structure=args.w_structure,
    )

    base_metric = v_l2_per_pos(alpha_base.detach().to(torch.float32), v_head, v_gt)
    linear_metric = v_l2_per_pos(alpha_linear.detach().to(torch.float32), v_head, v_gt)

    print("===== Compare-Q-Linear Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, tau_target={args.tau_target}, w_l2={args.w_l2:g}"
    )
    print(
        f"mean base v_l2={float(base_metric.mean().item()):.8e}, "
        f"mean linear v_l2={float(linear_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - linear_metric).mean().item()):.8e}"
    )

    per_pos_path = os.path.join(output_dir, "per_pos_v_l2.tsv")
    save_per_pos_metric_tsv(
        out_path=per_pos_path,
        pos_list=pos_list,
        base_metric=base_metric,
        other_metric=linear_metric,
        other_name="linear",
    )
    
    plot_path = os.path.join(output_dir, "per_pos_v_l2.png")
    plot_per_pos_two_lines(
        out_path=plot_path,
        pos_list=pos_list,
        y1=base_metric,
        y2=linear_metric,
        label1="base_v_l2",
        label2="linear_v_l2",
        title=f"Per-Position V-L2: Base vs Linear with budget={args.budget:g}",
        dpi=args.plot_dpi,
    )
    w_norm_tsv_path = os.path.join(output_dir, "w_norm_per_head.tsv")
    save_w_norm_tsv(w_norm_tsv_path, w, head_idx)

    w_norm_png_path = os.path.join(output_dir, "w_norm_per_head.png")
    plot_w_norm(w_norm_png_path, w, head_idx, dpi=args.plot_dpi)

    w_path = os.path.join(output_dir, "w_per_head.pt")
    torch.save(w, w_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "tau_target": args.tau_target,
        "metric_name": "v_l2",
        "w": w,
        "w_history": w_history,
        "mean_base_metric": float(base_metric.mean().item()),
        "mean_linear_metric": float(linear_metric.mean().item()),
        "mean_improvement": float((base_metric - linear_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "linear_metric_per_pos": linear_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_q_linear_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos v_l2 table to: {per_pos_path}")
    print(f"Saved W-norm table to: {w_norm_tsv_path}")
    print(f"Saved W-norm figure to: {w_norm_png_path}")
    print(f"Saved W tensor to: {w_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
