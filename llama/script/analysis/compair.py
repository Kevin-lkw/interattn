import argparse
import os

import matplotlib.pyplot as plt
import torch

from .attention import (
    build_qk_routing_alpha,
    gen_mask,
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
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Single head index to analyze. If omitted and --heads not set, run all heads.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="A list of head indices to analyze. Overrides --head.",
    )
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
    parser.add_argument(
        "--alpha-viz",
        type=str,
        default="log",
        choices=["linear", "log", "row_log"],
        help=(
            "Visualization scale for baseline/optimal alpha heatmaps. "
            "linear: raw probs; log: log10(probs); row_log: log10(probs / row_max)."
        ),
    )
    parser.add_argument(
        "--diff-log-eps",
        type=float,
        default=1e-6,
        help="Scale in signed-log diff map: sign(x)*log10(1 + |x|/eps).",
    )
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


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        if args.heads is not None and len(args.heads) > 0:
            head_tag = f"heads_{len(set(args.heads))}"
        elif args.head is not None:
            head_tag = f"head{args.head}"
        else:
            head_tag = "heads_all"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/compare/layer{args.layer}_{head_tag}/budget_{args.budget:g}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def resolve_head_indices(args, num_heads):
    if args.heads is not None and len(args.heads) > 0:
        return sorted(set(int(x) for x in args.heads))
    if args.head is not None:
        return [int(args.head)]
    return list(range(num_heads))


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
    # alpha_*: [n_heads, n_pos, seq_len]
    per_head = {}
    per_head_summaries = {}

    for head_idx in range(alpha_opt.shape[0]):
        per_pos = []
        for row_i, pos in enumerate(pos_list):
            total_available = pos + 1
            k = min(topk, total_available)
            if k <= 0:
                continue

            opt_row = alpha_opt[head_idx, row_i, :total_available]
            base_row = alpha_base[head_idx, row_i, :total_available]

            opt_idx = torch.topk(opt_row, k=k, largest=True).indices
            base_idx = torch.topk(base_row, k=k, largest=True).indices

            opt_set = set(opt_idx.detach().cpu().tolist())
            base_set = set(base_idx.detach().cpu().tolist())
            inter = opt_set.intersection(base_set)
            union = opt_set.union(base_set)

            overlap_ratio = len(inter) / float(k)
            jaccard = len(inter) / float(len(union)) if len(union) > 0 else 1.0

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
        per_head[head_idx] = per_pos
        per_head_summaries[head_idx] = summary

    mean_overlap = torch.tensor(
        [v["mean_overlap_ratio"] for v in per_head_summaries.values()], dtype=torch.float32
    )
    mean_jaccard = torch.tensor(
        [v["mean_jaccard"] for v in per_head_summaries.values()], dtype=torch.float32
    )
    global_summary = {
        "mean_overlap_ratio": mean_overlap.mean().item(),
        "std_overlap_ratio": mean_overlap.std(unbiased=False).item(),
        "mean_jaccard": mean_jaccard.mean().item(),
        "std_jaccard": mean_jaccard.std(unbiased=False).item(),
        "num_heads": len(per_head_summaries),
    }
    return global_summary, per_head_summaries, per_head


def _to_plot_array(x: torch.Tensor):
    x = x.detach().float().cpu().clone()
    finite = torch.isfinite(x)
    if finite.any():
        min_val = x[finite].min().item()
        x[~finite] = min_val - 1.0
    else:
        x[:] = 0.0
    return x.numpy()


def build_prob_viz_map(probs_tensor: torch.Tensor, mode: str, eps: float = 1e-8):
    probs = probs_tensor.detach().float()
    if mode == "linear":
        return probs, "linear", "Reds", 0.0

    if mode == "log":
        # Log scale reveals tail structure in highly peaked rows.
        z = torch.log10(probs.clamp_min(eps))
        return z, f"log10 (eps={eps:g})", "viridis", None

    if mode == "row_log":
        # Normalize each row by its max before log; highlights relative pattern per query position.
        row_max = probs.max(dim=-1, keepdim=True).values.clamp_min(eps)
        z = torch.log10((probs / row_max).clamp_min(eps))
        return z, f"row-normalized log10 (eps={eps:g})", "viridis", None

    raise ValueError(f"Unknown viz mode: {mode}")


def signed_log_map(x: torch.Tensor, eps: float):
    return torch.sign(x) * torch.log10(1.0 + x.abs() / eps)


def plot_routing_grid(alpha_base, alpha_opt, head_labels, out_path, dpi, alpha_viz, diff_log_eps):
    # alpha_*: [n_heads, n_pos, seq_len]
    n_heads = alpha_base.shape[0]
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 3, figsize=(15.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 3)

    for h in range(n_heads):
        base_map, alpha_mode_name, alpha_cmap, alpha_vmin = build_prob_viz_map(
            alpha_base[h], alpha_viz
        )
        opt_map, _alpha_mode_name2, _alpha_cmap2, _alpha_vmin2 = build_prob_viz_map(
            alpha_opt[h], alpha_viz
        )
        head_label = head_labels[h]

        diff_map = signed_log_map(alpha_opt[h] - alpha_base[h], diff_log_eps)
        diff_lim = max(diff_map.abs().max().item(), 1e-8)

        im0 = axes[h, 0].imshow(
            _to_plot_array(base_map),
            aspect="auto",
            cmap=alpha_cmap,
            vmin=alpha_vmin,
        )
        axes[h, 0].set_title(f"Head {head_label}: baseline ({alpha_mode_name})")
        fig.colorbar(im0, ax=axes[h, 0], fraction=0.046)

        im1 = axes[h, 1].imshow(
            _to_plot_array(opt_map),
            aspect="auto",
            cmap=alpha_cmap,
            vmin=alpha_vmin,
        )
        axes[h, 1].set_title(f"Head {head_label}: optimal ({alpha_mode_name})")
        fig.colorbar(im1, ax=axes[h, 1], fraction=0.046)

        im2 = axes[h, 2].imshow(
            _to_plot_array(diff_map),
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_lim,
            vmax=diff_lim,
        )
        axes[h, 2].set_title(f"Head {head_label}: diff signed-log (eps={diff_log_eps:g})")
        fig.colorbar(im2, ax=axes[h, 2], fraction=0.046)

        for col in range(3):
            axes[h, col].set_xlabel("Key position")
            axes[h, col].set_ylabel("Query position")

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
    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)

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
        head_idx=head_idx,
        strategy=args.strategy,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
    )

    alpha_baseline = build_qk_routing_alpha(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        mask=mask,
        device=ctx.device,
    )

    alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
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

    visible = int(args.seq_len * args.budget)
    topk = visible if args.topk is None else args.topk
    if topk <= 0:
        raise ValueError(
            f"topk={topk} is invalid. Increase --budget or provide a positive --topk."
        )

    summary, per_head_summaries, per_head = compute_overlap_stats(
        alpha_opt=alpha_opt,
        alpha_base=alpha_baseline,
        pos_list=pos_list,
        topk=topk,
    )

    mat_path = os.path.join(output_dir, "routing_grid.png")
    overlap_curve_path = os.path.join(output_dir, "topk_overlap_curve.png")
    stats_path = os.path.join(output_dir, "overlap_stats.pt")

    plot_routing_grid(
        alpha_base=alpha_baseline,
        alpha_opt=alpha_opt,
        head_labels=head_idx,
        out_path=mat_path,
        dpi=args.dpi,
        alpha_viz=args.alpha_viz,
        diff_log_eps=args.diff_log_eps,
    )

    # Keep overlap curve for the first selected head to avoid overly dense plots.
    first_head_local = 0
    plot_overlap_curve(
        per_pos=per_head[first_head_local],
        out_path=overlap_curve_path,
        dpi=args.dpi,
    )

    torch.save(
        {
            "summary": summary,
            "per_head_summary": per_head_summaries,
            "per_head": per_head,
            "layer": args.layer,
            "heads": head_idx,
            "budget": float(args.budget),
            "topk": int(topk),
            "prefix_mode": args.prefix_mode,
            "loss": losses,
        },
        stats_path,
    )

    base_row_sum_err = (alpha_baseline.sum(dim=-1) - 1.0).abs().max().item()
    opt_row_sum_err = (alpha_opt.sum(dim=-1) - 1.0).abs().max().item()

    print("===== Compare Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"topk={topk}, prefix_mode={args.prefix_mode}"
    )
    print(
        f"mean overlap ratio={summary['mean_overlap_ratio']:.6f} +- {summary['std_overlap_ratio']:.6f}"
    )
    print(f"mean jaccard={summary['mean_jaccard']:.6f} +- {summary['std_jaccard']:.6f}")
    print(f"num_heads={summary['num_heads']}")
    print(
        "max row-sum error: "
        f"baseline_alpha={base_row_sum_err:.3e}, optimal_alpha={opt_row_sum_err:.3e}"
    )
    print(f"saved matrix plot: {mat_path}")
    print(f"saved overlap curve: {overlap_curve_path}")
    print(f"saved stats: {stats_path}")


if __name__ == "__main__":
    main()