"""
Approximate optimal routing with per-key-column bias.

For each head h and key column j, routing logits are modified from qikj to qikj + Mhj.

Workflow:
1) Build prefix patches for layers < target layer.
2) On target layer, build baseline routing alpha from original QK logits.
3) Optimize alpha_opt (same as existing pipeline, via optimize_alpha_star).
4) Fit bias matrix M (shape [n_heads, seq_len]) by minimizing CE(alpha_opt, alpha_bias).
5) Print per-position KL(alpha_opt || alpha_base) and KL(alpha_opt || alpha_bias).
"""

import argparse
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
        description="Compare QK routing and per-column bias routing vs optimal routing."
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

    parser.add_argument("--bias-steps", type=int, default=200)
    parser.add_argument("--bias-lr", type=float, default=5e-2)
    parser.add_argument(
        "--bias-l2",
        type=float,
        default=0.0,
        help="Optional L2 regularization on centered bias vector.",
    )
    parser.add_argument("--bias-plot-dpi", type=int, default=180)

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
            f"{args.loss_type}/comprare_q_bias/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
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


def build_bias_routing_alpha(qk_logits, mask, bias):
    if bias.ndim != 2 or bias.shape != (qk_logits.shape[0], qk_logits.shape[-1]):
        raise ValueError(
            f"bias shape must be [n_heads, seq_len], got {tuple(bias.shape)} for "
            f"n_heads={qk_logits.shape[0]}, seq_len={qk_logits.shape[-1]}"
        )
    logits = qk_logits + bias.to(device=qk_logits.device, dtype=torch.float32).unsqueeze(1)
    return F.softmax(logits + mask.to(torch.float32), dim=-1)


def optimize_bias_ce(alpha_target, qk_logits, mask, bias_steps, bias_lr, bias_l2):
    n_heads = qk_logits.shape[0]
    seq_len = qk_logits.shape[-1]
    bias_param = torch.nn.Parameter(
        torch.zeros(n_heads, seq_len, dtype=torch.float32, device=qk_logits.device)
    )
    opt = torch.optim.Adam([bias_param], lr=bias_lr)

    alpha_target = alpha_target.to(torch.float32)
    mask_f = mask.to(torch.float32)
    eps = 1e-12
    history = []

    for step in range(int(bias_steps)):
        # Remove per-head global shift ambiguity in softmax logits.
        bias_centered = bias_param - bias_param.mean(dim=-1, keepdim=True)
        logits = qk_logits + bias_centered.unsqueeze(1) + mask_f
        logp = F.log_softmax(logits, dim=-1)

        target_row_sum = alpha_target.sum(dim=-1)
        valid_rows = (target_row_sum > eps) & torch.isfinite(logp).all(dim=-1)
        if not torch.any(valid_rows):
            raise ValueError("No valid rows for bias optimization. Check budget/mask settings.")

        alpha_target_norm = alpha_target / target_row_sum.clamp_min(eps).unsqueeze(-1)
        ce_rows = -(alpha_target_norm * logp).sum(dim=-1)
        ce = ce_rows[valid_rows].mean()

        if float(bias_l2) > 0.0:
            ce = ce + float(bias_l2) * (bias_centered.pow(2).mean())

        if not torch.isfinite(ce):
            raise ValueError("Bias optimization CE became non-finite.")

        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_([bias_param], max_norm=1.0)
        opt.step()

        if step % 20 == 0 or step == int(bias_steps) - 1:
            history.append((int(step), float(ce.detach().cpu().item())))
            print(f"[bias-opt] step={step:4d} ce={float(ce.detach().cpu().item()):.8f}")

    with torch.no_grad():
        final_bias = (bias_param - bias_param.mean()).detach().cpu()
        alpha_bias = build_bias_routing_alpha(qk_logits, mask, final_bias)

    return final_bias, alpha_bias, history


def kl_per_pos(alpha_target, alpha_pred):
    eps = 1e-12
    kl = (
        alpha_target
        * (torch.log(alpha_target.clamp_min(eps)) - torch.log(alpha_pred.clamp_min(eps)))
    ).sum(dim=-1)
    return kl.mean(dim=0)


def save_per_pos_kl_tsv(out_path, pos_list, kl_base, kl_bias):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("pos\tbase_kl\tbias_kl\tdelta_base_minus_bias\n")
        for i, pos in enumerate(pos_list):
            kb = float(kl_base[i].item())
            km = float(kl_bias[i].item())
            f.write(f"{pos}\t{kb:.8e}\t{km:.8e}\t{(kb - km):.8e}\n")


def save_bias_per_key_tsv(out_path, bias):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("head\tkey_idx\tbias\n")
        for h in range(int(bias.shape[0])):
            for j in range(int(bias.shape[1])):
                f.write(f"{h}\t{j}\t{float(bias[h, j].item()):.8e}\n")


def plot_bias_per_key(out_path, bias, dpi=180):
    x = list(range(int(bias.shape[1])))

    plt.figure(figsize=(10, 4.8))
    for h in range(int(bias.shape[0])):
        y = [float(v) for v in bias[h].tolist()]
        plt.plot(x, y, linewidth=1.0, alpha=0.8, label=f"h{h}")
    plt.xlabel("key column j")
    plt.ylabel("bias Mhj")
    plt.title("Learned Per-Head Per-Column Bias")
    plt.grid(alpha=0.25)
    if int(bias.shape[0]) <= 16:
        plt.legend(ncol=2, fontsize=8)
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

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    bias, alpha_bias, bias_history = optimize_bias_ce(
        alpha_target=alpha_opt.detach().to(torch.float32),
        qk_logits=qk_logits,
        mask=mask,
        bias_steps=args.bias_steps,
        bias_lr=args.bias_lr,
        bias_l2=args.bias_l2,
    )

    kl_base = kl_per_pos(alpha_opt.detach().to(torch.float32), alpha_base.detach().to(torch.float32))
    kl_bias = kl_per_pos(alpha_opt.detach().to(torch.float32), alpha_bias.detach().to(torch.float32))

    print("===== Compare-Q-Bias Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, bias_l2={args.bias_l2:g}"
    )
    print(
        f"mean base KL={float(kl_base.mean().item()):.8e}, "
        f"mean bias KL={float(kl_bias.mean().item()):.8e}, "
        f"mean improvement={float((kl_base - kl_bias).mean().item()):.8e}"
    )

    print("===== Per-Position KL (target: optimal alpha) =====")
    print("pos\tbase_kl\tbias_kl\tdelta(base-bias)")
    for i, pos in enumerate(pos_list):
        kb = float(kl_base[i].item())
        km = float(kl_bias[i].item())
        print(f"{pos}\t{kb:.8e}\t{km:.8e}\t{(kb - km):.8e}")

    per_pos_path = os.path.join(output_dir, "per_pos_kl.tsv")
    save_per_pos_kl_tsv(per_pos_path, pos_list, kl_base, kl_bias)

    bias_tsv_path = os.path.join(output_dir, "bias_per_key.tsv")
    save_bias_per_key_tsv(bias_tsv_path, bias)

    bias_png_path = os.path.join(output_dir, "bias_per_key.png")
    plot_bias_per_key(bias_png_path, bias, dpi=args.bias_plot_dpi)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "bias": bias,
        "bias_history": bias_history,
        "mean_base_kl": float(kl_base.mean().item()),
        "mean_bias_kl": float(kl_bias.mean().item()),
        "mean_improvement": float((kl_base - kl_bias).mean().item()),
        "kl_base_per_pos": kl_base.detach().cpu(),
        "kl_bias_per_pos": kl_bias.detach().cpu(),
    }
    stats_path = os.path.join(output_dir, "comprare_q_bias_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos KL table to: {per_pos_path}")
    print(f"Saved bias table to: {bias_tsv_path}")
    print(f"Saved bias figure to: {bias_png_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
