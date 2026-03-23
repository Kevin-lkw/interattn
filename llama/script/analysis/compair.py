import argparse
import os

import matplotlib.pyplot as plt
import torch

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
        description="Compare optimal routing and baseline routing for one layer/head."
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
    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
    )
    parser.add_argument("--adaptive-budget", action="store_true")

    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)

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

    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k used for overlap. Default: visible=int(seq_len*budget)",
    )
    parser.add_argument(
        "--pos-start",
        type=int,
        default=0,
        help="Start position (inclusive) for analysis.",
    )
    parser.add_argument(
        "--pos-end",
        type=int,
        default=None,
        help="End position (exclusive) for analysis. Default: seq_len.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads):
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.head < 0 or args.head >= num_heads:
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/compare/layer{args.layer}_head{args.head}/budget_{args.budget:g}"
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


def compute_overlap_stats(alpha_opt, alpha_base, pos_list, topk):
    per_pos = []
    for row_i, pos in enumerate(pos_list):
        total_available = pos + 1
        k = min(topk, total_available)
        if k <= 0:
            continue

        opt_row = alpha_opt[row_i, :total_available]
        base_row = alpha_base[row_i, :total_available]

        opt_idx = torch.topk(opt_row, k=k, largest=True).indices
        base_idx = torch.topk(base_row, k=k, largest=True).indices

        opt_set = set(opt_idx.detach().cpu().tolist())
        base_set = set(base_idx.detach().cpu().tolist())
        inter = opt_set.intersection(base_set)
        union = opt_set.union(base_set)

        overlap_ratio = len(inter) / float(k)
        jaccard = len(inter) / float(len(union)) if len(union) > 0 else 1.0

        # How much mass each routing puts on the other's top-k set.
        cross_mass_opt_on_base = opt_row[base_idx].sum().item()
        cross_mass_base_on_opt = base_row[opt_idx].sum().item()

        per_pos.append(
            {
                "pos": int(pos),
                "k": int(k),
                "overlap_ratio": float(overlap_ratio),
                "jaccard": float(jaccard),
                "opt_mass_on_base_topk": float(cross_mass_opt_on_base),
                "base_mass_on_opt_topk": float(cross_mass_base_on_opt),
            }
        )

    if len(per_pos) == 0:
        raise ValueError("No valid position for overlap statistics.")

    overlap_values = torch.tensor([x["overlap_ratio"] for x in per_pos], dtype=torch.float32)
    jaccard_values = torch.tensor([x["jaccard"] for x in per_pos], dtype=torch.float32)
    opt_on_base_values = torch.tensor(
        [x["opt_mass_on_base_topk"] for x in per_pos], dtype=torch.float32
    )
    base_on_opt_values = torch.tensor(
        [x["base_mass_on_opt_topk"] for x in per_pos], dtype=torch.float32
    )

    summary = {
        "mean_overlap_ratio": overlap_values.mean().item(),
        "std_overlap_ratio": overlap_values.std(unbiased=False).item(),
        "mean_jaccard": jaccard_values.mean().item(),
        "std_jaccard": jaccard_values.std(unbiased=False).item(),
        "mean_opt_mass_on_base_topk": opt_on_base_values.mean().item(),
        "mean_base_mass_on_opt_topk": base_on_opt_values.mean().item(),
        "num_positions": len(per_pos),
    }
    return summary, per_pos


def _to_plot_array(x: torch.Tensor):
    x = x.detach().float().cpu().clone()
    finite = torch.isfinite(x)
    if finite.any():
        min_val = x[finite].min().item()
        x[~finite] = min_val - 1.0
    else:
        x[:] = 0.0
    return x.numpy()


def plot_routing_matrices(qk_probs_head, alpha_base, alpha_opt, out_path, dpi):
    diff = alpha_opt - alpha_base

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)

    im0 = axes[0, 0].imshow(_to_plot_array(qk_probs_head), aspect="auto", cmap="Reds", vmin=0.0)
    axes[0, 0].set_title("QK routing prob (causal softmax)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(_to_plot_array(alpha_base), aspect="auto", cmap="Reds", vmin=0.0)
    axes[0, 1].set_title("Baseline routing alpha")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(_to_plot_array(alpha_opt), aspect="auto", cmap="Reds", vmin=0.0)
    axes[1, 0].set_title("Optimal routing alpha")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    max_abs = max(diff.abs().max().item(), 1e-8)
    im3 = axes[1, 1].imshow(
        _to_plot_array(diff),
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axes[1, 1].set_title("Alpha diff (optimal - baseline)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_overlap_curve(per_pos, out_path, dpi):
    positions = [x["pos"] for x in per_pos]
    overlap = [x["overlap_ratio"] for x in per_pos]
    jaccard = [x["jaccard"] for x in per_pos]
    mass1 = [x["opt_mass_on_base_topk"] for x in per_pos]
    mass2 = [x["base_mass_on_opt_topk"] for x in per_pos]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), constrained_layout=True)

    axes[0].plot(positions, overlap, label="top-k overlap ratio", linewidth=1.4)
    axes[0].plot(positions, jaccard, label="jaccard", linewidth=1.2)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Set overlap")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(positions, mass1, label="opt mass on base top-k", linewidth=1.4)
    axes[1].plot(positions, mass2, label="base mass on opt top-k", linewidth=1.4)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Cross mass")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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

    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty pos_list after applying --pos-start/--pos-end")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(args)

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
        head_idx=[args.head],
        strategy=args.strategy,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
    )

    alpha_baseline = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=[args.head],
        pos_list=pos_list,
        mask=mask,
        device=ctx.device,
    )[0]

    alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=[args.head],
        pos_list=pos_list,
        training_steps=args.training_steps,
        lr=args.lr,
        mask=mask,
        loss_type=args.loss_type,
        device=ctx.device,
    )
    alpha_opt = alpha_opt[0]

    scores, attn = get_attention_map_after_rope(
        ctx=layer_ctx,
        layer_idx=args.layer,
        causal=True,
        dtype=torch.float32,
        device=ctx.device,
    )
    qk_scores_head = scores[args.head][pos_list, :]
    qk_probs_head = attn[args.head][pos_list, :]

    visible = int(args.seq_len * args.budget)
    topk = visible if args.topk is None else args.topk
    if topk <= 0:
        raise ValueError(
            f"topk={topk} is invalid. Increase --budget or provide a positive --topk."
        )

    summary, per_pos = compute_overlap_stats(
        alpha_opt=alpha_opt,
        alpha_base=alpha_baseline,
        pos_list=pos_list,
        topk=topk,
    )

    mat_path = os.path.join(output_dir, "routing_matrices.png")
    overlap_curve_path = os.path.join(output_dir, "topk_overlap_curve.png")
    stats_path = os.path.join(output_dir, "overlap_stats.pt")
    alpha_dump_path = os.path.join(output_dir, "routing_tensors.pt")

    plot_routing_matrices(
        qk_probs_head=qk_probs_head,
        alpha_base=alpha_baseline,
        alpha_opt=alpha_opt,
        out_path=mat_path,
        dpi=args.dpi,
    )
    plot_overlap_curve(per_pos=per_pos, out_path=overlap_curve_path, dpi=args.dpi)

    torch.save(
        {
            "summary": summary,
            "per_pos": per_pos,
            "layer": args.layer,
            "head": args.head,
            "budget": float(args.budget),
            "topk": int(topk),
            "prefix_mode": args.prefix_mode,
            "loss": losses,
        },
        stats_path,
    )
    torch.save(
        {
            "qk_scores_head": qk_scores_head.detach().cpu(),
            "qk_probs_head": qk_probs_head.detach().cpu(),
            "alpha_baseline": alpha_baseline.detach().cpu(),
            "alpha_optimal": alpha_opt.detach().cpu(),
            "mask": mask[0].detach().cpu(),
            "pos_list": pos_list,
        },
        alpha_dump_path,
    )

    qk_row_sum_err = (qk_probs_head.sum(dim=-1) - 1.0).abs().max().item()
    base_row_sum_err = (alpha_baseline.sum(dim=-1) - 1.0).abs().max().item()
    opt_row_sum_err = (alpha_opt.sum(dim=-1) - 1.0).abs().max().item()

    print("===== Compare Summary =====")
    print(
        f"layer={args.layer}, head={args.head}, budget={args.budget:g}, "
        f"topk={topk}, prefix_mode={args.prefix_mode}"
    )
    print(
        f"mean overlap ratio={summary['mean_overlap_ratio']:.6f} +- {summary['std_overlap_ratio']:.6f}"
    )
    print(f"mean jaccard={summary['mean_jaccard']:.6f} +- {summary['std_jaccard']:.6f}")
    print(f"mean opt_mass_on_base_topk={summary['mean_opt_mass_on_base_topk']:.6f}")
    print(f"mean base_mass_on_opt_topk={summary['mean_base_mass_on_opt_topk']:.6f}")
    print(
        "max row-sum error: "
        f"qk_prob={qk_row_sum_err:.3e}, baseline_alpha={base_row_sum_err:.3e}, optimal_alpha={opt_row_sum_err:.3e}"
    )
    print(f"saved matrix plot: {mat_path}")
    print(f"saved overlap curve: {overlap_curve_path}")
    print(f"saved stats: {stats_path}")
    print(f"saved tensors: {alpha_dump_path}")


if __name__ == "__main__":
    main()