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

from .attention import build_qk_routing_alpha, gen_mask
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import get_result_path, load_context, normalize_budget_key
from .sanity import build_modified_attn_hidden, move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare QK routing and Q-linear routing (qWk) with v_l2_gt objective."
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
    parser.add_argument("--plot-dpi", type=int, default=180)
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
            f"compare_q_linear/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
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
        loss_type="v_l2",
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


def save_per_pos_v_l2_tsv(out_path, pos_list, base_metric, linear_metric):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("pos\tbase_v_l2\tlinear_v_l2\tdelta_base_minus_linear\n")
        for i, pos in enumerate(pos_list):
            mb = float(base_metric[i].item())
            ml = float(linear_metric[i].item())
            f.write(f"{pos}\t{mb:.8e}\t{ml:.8e}\t{(mb - ml):.8e}\n")


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
    save_per_pos_v_l2_tsv(per_pos_path, pos_list, base_metric, linear_metric)

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
