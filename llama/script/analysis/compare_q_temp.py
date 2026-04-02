"""
Compare baseline QK routing and temperature-scaled QK routing against optimal routing.

Workflow:
1) Build prefix patches for layers < target layer.
2) On target layer, build baseline routing alpha from original QK logits.
3) Optimize alpha_opt (same as existing pipeline, via optimize_alpha_star).
4) Fit per-head temperature tau by minimizing CE(alpha_opt, alpha_tau).
5) Print per-position KL(alpha_opt || alpha_base) and KL(alpha_opt || alpha_tau).
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .attention import (
    build_qk_routing_alpha,
    gen_mask,
    get_attention_map_after_rope,
    optimize_alpha_star,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import get_result_path, load_context, normalize_budget_key
from .sanity import build_modified_attn_hidden, move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare QK routing and temperature-scaled QK routing vs optimal routing."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="h2o",
        choices=["recency", "random", "attention_topk", "h2o", "kvmerger", "sink"],
    )
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budget", type=float, required=True)

    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)

    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
        help="Loss used to optimize alpha_opt.",
    )
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument(
        "--tau-steps",
        type=int,
        default=200,
        help="Optimization steps for temperature tau.",
    )
    parser.add_argument("--tau-lr", type=float, default=5e-2)
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--tau-min", type=float, default=1e-3)
    parser.add_argument("--tau-max", type=float, default=1e3)
    parser.add_argument(
        "--tau-granularity",
        type=str,
        default="head",
        choices=["head", "head_query"],
        help=(
            "Granularity of tau. "
            "head: one tau per head; "
            "head_query: one tau per (head, query position)."
        ),
    )
    parser.add_argument(
        "--tau-target",
        type=str,
        default="optimal_ce",
        choices=["optimal_ce", "v_l2_gt"],
        help=(
            "Target for tau optimization. "
            "optimal_ce: CE(alpha_opt, alpha_tau); "
            "v_l2_gt: directly minimize ||V_tau - V_gt||_2 (skip optimal routing)."
        ),
    )
    parser.add_argument("--tau-plot-dpi", type=int, default=180)

    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="optimal_saved",
        choices=["optimal_saved", "baseline_rebuild"],
        help=(
            "How to prepare patches before target layer. "
            "optimal_saved: load saved optimal patch_hidden for layers < target; "
            "baseline_rebuild: rebuild baseline patches online for layers < target."
        ),
    )

    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.head is not None and (args.head < 0 or args.head >= num_heads):
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.heads is not None:
        for h in args.heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Invalid --heads entry {h}; expected [0, {num_heads - 1}]")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")
    if args.tau_init <= 0:
        raise ValueError("--tau-init must be > 0")
    if args.tau_min <= 0:
        raise ValueError("--tau-min must be > 0")
    if args.tau_max <= args.tau_min:
        raise ValueError("--tau-max must be larger than --tau-min")


def resolve_head_indices(args, num_heads):
    if args.heads is not None and len(args.heads) > 0:
        return sorted(set(int(x) for x in args.heads))
    if args.head is not None:
        return [int(args.head)]
    return list(range(num_heads))


def resolve_output_dir(args, head_idx):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        if len(head_idx) == 1:
            head_tag = f"head{head_idx[0]}"
        else:
            head_tag = f"heads_{len(head_idx)}"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/compare_q/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_saved_patch_hidden_for_layer(args, layer_idx, budget, device):
    path = get_result_path(
        layer_idx=layer_idx,
        dataset=args.dataset,
        start=args.start,
        adaptive_budget=args.adaptive_budget,
        strategy=args.strategy,
        loss_type=args.loss_type,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing optimal layer result for layer={layer_idx}. Expected file: {path}"
        )

    result = torch.load(path, map_location="cpu", weights_only=False)
    key = normalize_budget_key(result, budget)
    if key is None:
        raise KeyError(
            f"Budget {budget} not found in layer={layer_idx} result keys: {list(result.keys())}"
        )

    entry = result[key]
    if not isinstance(entry, dict) or "patch_hidden" not in entry:
        raise KeyError(
            f"layer={layer_idx}, budget={budget} has no patch_hidden in saved result entry."
        )
    return entry["patch_hidden"].to(device)


def build_optimal_saved_prefix_patches(args, target_layer, budget, device):
    patches = {}
    for layer_idx in range(target_layer):
        patches[layer_idx] = load_saved_patch_hidden_for_layer(args, layer_idx, budget, device)
    return patches


def build_baseline_prefix_patches(ctx, args, target_layer, pos_list, model_inputs):
    patches = {}
    head_idx = list(range(ctx.model_config.num_attention_heads))
    for layer_idx in range(target_layer):
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=patches,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        mask = gen_mask(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=head_idx,
            strategy=args.strategy,
            budget=args.budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )
        alpha_baseline = build_qk_routing_alpha(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            mask=mask,
            device=ctx.device,
        )
        patches[layer_idx] = build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            alpha=alpha_baseline,
            device=ctx.device,
        )
        print(f"[prefix baseline rebuild] layer {layer_idx} done")
    return patches


def get_qk_logits(ctx, layer_idx, head_idx, pos_list, device):
    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    return qk_scores[head_idx][:, pos_list, :].to(torch.float32)


def build_temp_scaled_alpha(qk_logits, mask, tau):
    if torch.is_tensor(tau):
        if tau.ndim == 1:
            if tau.shape[0] != qk_logits.shape[0]:
                raise ValueError(
                    f"tau tensor shape must be [n_heads], got {tuple(tau.shape)} for n_heads={qk_logits.shape[0]}"
                )
            tau_view = tau.to(device=qk_logits.device, dtype=torch.float32).view(-1, 1, 1)
        elif tau.ndim == 2:
            expected = (qk_logits.shape[0], qk_logits.shape[1])
            if tuple(tau.shape) != expected:
                raise ValueError(
                    f"tau tensor shape must be [n_heads, n_pos], got {tuple(tau.shape)}; expected {expected}"
                )
            tau_view = tau.to(device=qk_logits.device, dtype=torch.float32).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported tau ndim={tau.ndim}; expected 1 or 2")
        if not torch.all(tau > 0):
            raise ValueError("All tau values must be > 0")
    else:
        tau_val = float(tau)
        if tau_val <= 0:
            raise ValueError(f"tau must be > 0, got {tau_val}")
        tau_view = torch.tensor(tau_val, dtype=torch.float32, device=qk_logits.device).view(1, 1, 1)

    return F.softmax(qk_logits / tau_view + mask.to(torch.float32), dim=-1)


def optimize_tau_ce(
    alpha_target,
    qk_logits,
    mask,
    tau_init,
    tau_steps,
    tau_lr,
    tau_min,
    tau_max,
    tau_granularity="head",
):
    n_heads = qk_logits.shape[0]
    n_pos = qk_logits.shape[1]
    if tau_granularity == "head_query":
        tau_shape = (n_heads, n_pos)
    else:
        tau_shape = (n_heads,)

    log_tau = torch.nn.Parameter(
        torch.full(tau_shape, math.log(tau_init), dtype=torch.float32, device=qk_logits.device)
    )
    opt = torch.optim.Adam([log_tau], lr=tau_lr)

    history = []
    eps = 1e-12
    for step in range(tau_steps):
        tau = torch.exp(log_tau)
        tau = tau.clamp(min=tau_min, max=tau_max)
        tau.retain_grad()
        is_masked = torch.isneginf(mask) | (mask < -1e8)
        qk_safe = qk_logits.masked_fill(is_masked, 0.0)
        safe_mask = mask.to(torch.float32)
        safe_mask = torch.where(
            torch.isneginf(safe_mask),
            torch.full_like(safe_mask, -1e9),
            safe_mask,
        )
        if tau_granularity == "head_query":
            tau_view = tau.unsqueeze(-1)
        else:
            tau_view = tau.view(-1, 1, 1)
        logits = qk_safe / tau_view + safe_mask
        ce = -(alpha_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        ce.backward()
        opt.step()

        if step % 10 == 0 or step == tau_steps - 1:
            ce_v = float(ce.detach().cpu().item())
            tau_v = tau.detach().cpu().tolist()
            history.append((step, tau_v, ce_v))
            print(f"[tau-opt] step={step:4d} ce={ce_v:.8f}")

    with torch.no_grad():
        final_tau = torch.exp(log_tau).clamp(min=tau_min, max=tau_max).detach()
        alpha_scaled = build_temp_scaled_alpha(qk_logits, mask, final_tau)

    return final_tau.cpu(), alpha_scaled, history


def optimize_tau_v_l2(
    v_head,
    v_gt,
    qk_logits,
    mask,
    tau_init,
    tau_steps,
    tau_lr,
    tau_min,
    tau_max,
    tau_granularity="head",
):
    n_heads = qk_logits.shape[0]
    n_pos = qk_logits.shape[1]
    if tau_granularity == "head_query":
        tau_shape = (n_heads, n_pos)
    else:
        tau_shape = (n_heads,)

    log_tau = torch.nn.Parameter(
        torch.full(tau_shape, math.log(tau_init), dtype=torch.float32, device=qk_logits.device)
    )
    opt = torch.optim.Adam([log_tau], lr=tau_lr)

    history = []
    for step in range(tau_steps):
        tau = torch.exp(log_tau)
        tau = tau.clamp(min=tau_min, max=tau_max)

        is_masked = torch.isneginf(mask) | (mask < -1e8)
        qk_safe = qk_logits.masked_fill(is_masked, 0.0)
        safe_mask = mask.to(torch.float32)
        safe_mask = torch.where(
            torch.isneginf(safe_mask),
            torch.full_like(safe_mask, -1e9),
            safe_mask,
        )

        if tau_granularity == "head_query":
            tau_view = tau.unsqueeze(-1)
        else:
            tau_view = tau.view(-1, 1, 1)
        alpha_tau = F.softmax(qk_safe / tau_view + safe_mask, dim=-1)
        v_new = alpha_tau @ v_head.float()
        loss = torch.norm(v_new - v_gt.float(), p=2, dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == tau_steps - 1:
            loss_v = float(loss.detach().cpu().item())
            tau_v = tau.detach().cpu().tolist()
            history.append((step, tau_v, loss_v))
            print(f"[tau-opt-v] step={step:4d} loss={loss_v:.8f}")

    with torch.no_grad():
        final_tau = torch.exp(log_tau).clamp(min=tau_min, max=tau_max).detach()
        alpha_scaled = build_temp_scaled_alpha(qk_logits, mask, final_tau)

    return final_tau.cpu(), alpha_scaled, history


def kl_per_pos(alpha_target, alpha_pred):
    eps = 1e-12
    kl = (
        alpha_target * (
            torch.log(alpha_target.clamp_min(eps)) - torch.log(alpha_pred.clamp_min(eps))
        )
    ).sum(dim=-1)
    return kl.mean(dim=0)


def v_l2_per_pos(alpha, v_head, v_gt):
    v_new = alpha.float() @ v_head.float()
    l2 = torch.norm(v_new - v_gt.float(), p=2, dim=-1)
    return l2.mean(dim=0)


def save_per_pos_metric_tsv(out_path, pos_list, base_metric, scaled_metric, metric_name):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"pos\tbase_{metric_name}\tscaled_{metric_name}\tdelta_base_minus_scaled\n")
        for i, pos in enumerate(pos_list):
            mb = float(base_metric[i].item())
            ms = float(scaled_metric[i].item())
            f.write(f"{pos}\t{mb:.8e}\t{ms:.8e}\t{(mb - ms):.8e}\n")


def save_tau_per_head_tsv(out_path, tau_per_head, head_idx):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("head\ttau\n")
        for i, h in enumerate(head_idx):
            f.write(f"{h}\t{float(tau_per_head[i].item()):.8e}\n")


def save_tau_per_head_query_tsv(out_path, tau_per_hq, head_idx, pos_list):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("head\tpos\ttau\n")
        for i, h in enumerate(head_idx):
            for j, pos in enumerate(pos_list):
                f.write(f"{h}\t{pos}\t{float(tau_per_hq[i, j].item()):.8e}\n")


def plot_tau_per_head_points(out_path, tau_per_head, head_idx, dpi=180):
    if len(head_idx) == 0:
        return

    x = head_idx
    y = [float(tau_per_head[i].item()) for i in range(len(head_idx))]

    plt.figure(figsize=(9, 4.8))
    plt.scatter(x, y, s=30)

    plt.xlabel("head")
    plt.ylabel("tau")
    plt.title("Final Tau Per Head")
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

    validate_args(
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
    output_dir = resolve_output_dir(args, head_idx)

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
        )

    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    mask = gen_mask(
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
        mask=mask,
        device=ctx.device,
    )

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )
    alpha_opt = None
    metric_name = "kl"

    if args.tau_target == "optimal_ce":
        alpha_opt, _p_alpha, _p_teacher, _losses = optimize_alpha_star(
            ctx=layer_ctx,
            layer_idx=args.layer,
            head_idx=head_idx,
            pos_list=pos_list,
            training_steps=args.training_steps,
            lr=args.lr,
            mask=mask,
            loss_type=args.loss_type,
            device=ctx.device,
        )

        tau_param, alpha_scaled, tau_history = optimize_tau_ce(
            alpha_target=alpha_opt.detach().to(torch.float32),
            qk_logits=qk_logits,
            mask=mask,
            tau_init=args.tau_init,
            tau_steps=args.tau_steps,
            tau_lr=args.tau_lr,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_granularity=args.tau_granularity,
        )

        base_metric = kl_per_pos(alpha_opt.detach().to(torch.float32), alpha_base.detach().to(torch.float32))
        scaled_metric = kl_per_pos(alpha_opt.detach().to(torch.float32), alpha_scaled.detach().to(torch.float32))
    else:
        v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
        v_gt = (
            layer_ctx.attn_output[args.layer]["output"][0, pos_list]
            .permute(1, 0, 2)
            .to(ctx.device)[head_idx]
            .float()
        )

        tau_param, alpha_scaled, tau_history = optimize_tau_v_l2(
            v_head=v_head,
            v_gt=v_gt,
            qk_logits=qk_logits,
            mask=mask,
            tau_init=args.tau_init,
            tau_steps=args.tau_steps,
            tau_lr=args.tau_lr,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_granularity=args.tau_granularity,
        )

        metric_name = "v_l2"
        base_metric = v_l2_per_pos(alpha_base.detach().to(torch.float32), v_head, v_gt)
        scaled_metric = v_l2_per_pos(alpha_scaled.detach().to(torch.float32), v_head, v_gt)

    print("===== Compare-Q Summary =====")
    if tau_param.ndim == 1:
        tau_per_head = tau_param
    else:
        tau_per_head = tau_param.mean(dim=1)
    tau_pairs = ", ".join(f"h{h}:{float(tau_per_head[i].item()):.6f}" for i, h in enumerate(head_idx))
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, tau_target={args.tau_target}, "
        f"tau_granularity={args.tau_granularity}, tau_per_head=[{tau_pairs}]"
    )
    print(
        f"mean base {metric_name}={float(base_metric.mean().item()):.8e}, "
        f"mean scaled {metric_name}={float(scaled_metric.mean().item()):.8e}, "
        f"mean improvement={float((base_metric - scaled_metric).mean().item()):.8e}"
    )


    per_pos_path = os.path.join(output_dir, f"per_pos_{metric_name}.tsv")
    save_per_pos_metric_tsv(per_pos_path, pos_list, base_metric, scaled_metric, metric_name)

    tau_points_tsv_path = os.path.join(output_dir, "tau_per_head.tsv")
    save_tau_per_head_tsv(tau_points_tsv_path, tau_per_head, head_idx)

    tau_hq_tsv_path = None
    if tau_param.ndim == 2:
        tau_hq_tsv_path = os.path.join(output_dir, "tau_per_head_query.tsv")
        save_tau_per_head_query_tsv(tau_hq_tsv_path, tau_param, head_idx, pos_list)

    tau_points_png_path = os.path.join(output_dir, "tau_per_head.png")
    plot_tau_per_head_points(
        out_path=tau_points_png_path,
        tau_per_head=tau_per_head,
        head_idx=head_idx,
        dpi=args.tau_plot_dpi,
    )

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "tau_target": args.tau_target,
        "tau_granularity": args.tau_granularity,
        "metric_name": metric_name,
        "tau_per_head": {h: float(tau_per_head[i].item()) for i, h in enumerate(head_idx)},
        "tau_param": tau_param,
        "tau_history": tau_history,
        "mean_base_metric": float(base_metric.mean().item()),
        "mean_scaled_metric": float(scaled_metric.mean().item()),
        "mean_improvement": float((base_metric - scaled_metric).mean().item()),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "scaled_metric_per_pos": scaled_metric.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "compare_q_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos {metric_name} table to: {per_pos_path}")
    print(f"Saved tau-per-head table to: {tau_points_tsv_path}")
    if tau_hq_tsv_path is not None:
        print(f"Saved tau-per-head-query table to: {tau_hq_tsv_path}")
    print(f"Saved tau-per-head figure to: {tau_points_png_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
