import argparse
import os

import matplotlib.pyplot as plt
import torch

from .attention import build_qk_routing_alpha, gen_mask, optimize_alpha_star
from .compair import (
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For each layer/head, interpolate baseline routing to optimal routing and "
            "plot ||V - V*|| curves."
        )
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

    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads):
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"Invalid --layer {args.layer}; expected [0, {num_layers - 1}]")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")

    if args.head is not None and (args.head < 0 or args.head >= num_heads):
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.heads is not None:
        for h in args.heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Invalid --heads entry {h}; expected [0, {num_heads - 1}]")


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        layer_tag = f"layers_{args.layer}"

        if args.heads is not None and len(args.heads) > 0:
            head_tag = f"heads_{len(set(args.heads))}"
        elif args.head is not None:
            head_tag = f"head{args.head}"
        else:
            head_tag = "heads_all"

        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/inter/{layer_tag}_{head_tag}/budget_{args.budget:g}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_rank_orders_for_one_row(delta_row, v_abs_row):
    # delta_row, v_abs_row: [k_valid]
    score_abs = delta_row.abs()
    score_abs_v = score_abs * v_abs_row

    order_abs = torch.argsort(score_abs, descending=True)
    order_abs_v = torch.argsort(score_abs_v, descending=True)
    return order_abs, order_abs_v


def _build_inter_curve_for_one_head(alpha_base_h, alpha_opt_h, v_h, v_abs_h, valid_indices_per_row):
    """
    alpha_*_h: [n_pos, seq_len]
    v_h: [seq_len, head_dim]
    v_abs_h: [seq_len]
    valid_indices_per_row: list[Tensor], each tensor contains visible key indices for one query row

        Return:
            n_steps: int (max number of replacement operations)
      curve_abs: list[float]       sorted by |delta alpha|
      curve_abs_v: list[float]     sorted by |delta alpha|*|V|
    """
    n_pos = alpha_base_h.shape[0]
    n_steps = int(max(int(idx.numel()) for idx in valid_indices_per_row))
    if n_steps <= 0:
        raise ValueError("No valid routing entries for interpolation.")

    per_pos_abs = []
    per_pos_abs_v = []

    for row_i in range(n_pos):
        valid_idx = valid_indices_per_row[row_i]
        k_valid = int(valid_idx.numel())
        if k_valid <= 0:
            per_pos_abs.append([0.0] * n_steps)
            per_pos_abs_v.append([0.0] * n_steps)
            continue

        base_row = alpha_base_h[row_i, valid_idx].detach().float()
        opt_row = alpha_opt_h[row_i, valid_idx].detach().float()

        v_valid = v_h[valid_idx].detach().float()
        v_abs_valid = v_abs_h[valid_idx].detach().float()

        delta_row = opt_row - base_row
        order_abs, order_abs_v = _build_rank_orders_for_one_row(delta_row, v_abs_valid)

        v_star = opt_row @ v_valid

        row_curve_abs = []
        row_curve_abs_v = []

        mix_abs = base_row.clone()
        mix_abs_v = base_row.clone()

        # t=0: replace nothing.
        v_now_abs0 = mix_abs @ v_valid
        v_now_abs_v0 = mix_abs_v @ v_valid
        row_curve_abs.append(float(torch.norm(v_now_abs0 - v_star, p=2).item()))
        row_curve_abs_v.append(float(torch.norm(v_now_abs_v0 - v_star, p=2).item()))

        for t in range(1, n_steps + 1):
            if t <= k_valid:
                idx_a = int(order_abs[t - 1].item())
                idx_b = int(order_abs_v[t - 1].item())
                mix_abs[idx_a] = opt_row[idx_a]
                mix_abs_v[idx_b] = opt_row[idx_b]

                v_now_abs = mix_abs @ v_valid
                v_now_abs_v = mix_abs_v @ v_valid

                err_abs = torch.norm(v_now_abs - v_star, p=2).item()
                err_abs_v = torch.norm(v_now_abs_v - v_star, p=2).item()
            else:
                # All valid entries are already replaced.
                err_abs = 0.0
                err_abs_v = 0.0

            row_curve_abs.append(float(err_abs))
            row_curve_abs_v.append(float(err_abs_v))

        per_pos_abs.append(row_curve_abs)
        per_pos_abs_v.append(row_curve_abs_v)

    curve_abs = torch.tensor(per_pos_abs, dtype=torch.float32).mean(dim=0).tolist()
    curve_abs_v = torch.tensor(per_pos_abs_v, dtype=torch.float32).mean(dim=0).tolist()
    return n_steps, curve_abs, curve_abs_v


def compute_inter_curves_per_head(alpha_base, alpha_opt, v_selected, pos_list, mask):
    """
    alpha_*: [n_heads, n_pos, seq_len]
    v_selected: [n_heads, seq_len, head_dim]
    mask: [n_heads, n_pos, seq_len]

    Return dict keyed by local head index in alpha_* tensors.
    """
    if alpha_base.shape != alpha_opt.shape:
        raise ValueError(
            f"Shape mismatch: alpha_base={tuple(alpha_base.shape)} vs alpha_opt={tuple(alpha_opt.shape)}"
        )
    if v_selected.ndim != 3:
        raise ValueError(f"v_selected must be [n_heads, seq_len, head_dim], got {tuple(v_selected.shape)}")
    if mask.shape != alpha_base.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match alpha shape {tuple(alpha_base.shape)}")

    n_heads, n_pos, seq_len = alpha_base.shape
    if len(pos_list) != n_pos:
        raise ValueError(f"len(pos_list)={len(pos_list)} does not match n_pos={n_pos}")

    v_abs = torch.norm(v_selected.detach().float(), p=2, dim=-1)  # [n_heads, seq_len]
    out = {}

    for h in range(n_heads):
        valid_indices_per_row = []
        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            valid_mask = torch.isfinite(mask[h, row_i, :total_available])
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            valid_indices_per_row.append(valid_idx)

        n_steps, curve_abs, curve_abs_v = _build_inter_curve_for_one_head(
            alpha_base_h=alpha_base[h],
            alpha_opt_h=alpha_opt[h],
            v_h=v_selected[h],
            v_abs_h=v_abs[h],
            valid_indices_per_row=valid_indices_per_row,
        )

        out[h] = {
            "x": list(range(0, n_steps + 1)),
            "curve_abs_diff": curve_abs,
            "curve_abs_diff_mul_v": curve_abs_v,
            "n_steps": n_steps,
            "n_pos": n_pos,
        }

    return out


def plot_stacked_head_curves(curves_per_head, head_labels, layer_idx, out_path, dpi):
    n_heads = len(head_labels)
    fig_h = max(2.7 * n_heads, 5.0)
    fig, axes = plt.subplots(n_heads, 1, figsize=(9.5, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = [axes]

    for i, head_label in enumerate(head_labels):
        curve_info = curves_per_head[i]
        xs = curve_info["x"]
        ys_abs = curve_info["curve_abs_diff"]
        ys_abs_v = curve_info["curve_abs_diff_mul_v"]

        ax = axes[i]
        ax.plot(xs, ys_abs, label="replace by |delta alpha|", linewidth=1.8)
        ax.plot(xs, ys_abs_v, label="replace by |delta alpha|*|V|", linewidth=1.8)
        ax.set_xlim(left=0)
        ax.set_xlabel("# replaced routing entries")
        ax.set_ylabel("||V - V*||_2")
        ax.set_title(f"Layer {layer_idx} Head {head_label}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

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

    summary = {
        "dataset": args.dataset,
        "start": args.start,
        "seq_len": args.seq_len,
        "budget": float(args.budget),
        "layer": int(args.layer),
        "heads": head_idx,
        "prefix_mode": args.prefix_mode,
        "curves": {},
    }
    layer_idx = int(args.layer)
    print(f"===== Layer {layer_idx} =====")
    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=layer_idx,
            budget=args.budget,
            device=ctx.device,
        )
    else:
        prefix_patches = build_baseline_prefix_patches(
            ctx=ctx,
            args=args,
            target_layer=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=layer_idx,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
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

    alpha_opt, _p_alpha, _p_teacher, losses = optimize_alpha_star(
        ctx=layer_ctx,
        layer_idx=layer_idx,
        head_idx=head_idx,
        pos_list=pos_list,
        training_steps=args.training_steps,
        lr=args.lr,
        mask=mask,
        loss_type=args.loss_type,
        device=ctx.device,
    )

    v_selected = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()

    curves_per_head = compute_inter_curves_per_head(
        alpha_base=alpha_baseline,
        alpha_opt=alpha_opt,
        v_selected=v_selected,
        pos_list=pos_list,
        mask=mask,
    )

    layer_store = {
        "loss": losses,
        "curves_per_head": {},
    }

    for i, head_label in enumerate(head_idx):
        curve_info = curves_per_head[i]
        print(
            f"layer={layer_idx}, head={head_label}, points={curve_info['n_steps']}, "
            f"start(abs)={curve_info['curve_abs_diff'][0]:.6e}, "
            f"start(abs*|V|)={curve_info['curve_abs_diff_mul_v'][0]:.6e}, "
            f"final(abs)={curve_info['curve_abs_diff'][-1]:.6e}, "
            f"final(abs*|V|)={curve_info['curve_abs_diff_mul_v'][-1]:.6e}"
        )

        layer_store["curves_per_head"][head_label] = curve_info

    stacked_fig_path = os.path.join(output_dir, "inter_curves_stacked.png")
    plot_stacked_head_curves(
        curves_per_head=curves_per_head,
        head_labels=head_idx,
        layer_idx=layer_idx,
        out_path=stacked_fig_path,
        dpi=args.dpi,
    )
    print(f"saved plot: {stacked_fig_path}")

    summary["curves"][layer_idx] = layer_store

    stats_path = os.path.join(output_dir, "interpolation_stats.pt")
    torch.save(summary, stats_path)
    print(f"saved stats: {stats_path}")


if __name__ == "__main__":
    main()
