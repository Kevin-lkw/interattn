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
        default=1e-2,
        help="Scale in signed-log diff map: sign(x)*log10(1 + |x|/eps).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sparsity-thresholds",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4],
        help="Thresholds for density check: report fraction of weights above each threshold.",
    )
    parser.add_argument(
        "--sparsity-mass-levels",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Mass levels for top-k coverage ratio, e.g. 0.9 means minimal k covering 90% mass.",
    )
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


def summarize_signed_diff_topk_per_head(alpha_opt, alpha_base, pos_list, topk=10):
    """Per-head signed diff extremes on valid causal prefixes."""
    if alpha_opt.shape != alpha_base.shape:
        raise ValueError(
            f"Shape mismatch: alpha_opt={tuple(alpha_opt.shape)} vs alpha_base={tuple(alpha_base.shape)}"
        )
    if topk <= 0:
        raise ValueError("topk must be > 0")

    diff = (alpha_opt - alpha_base).detach().float()
    n_heads = diff.shape[0]
    n_pos = diff.shape[1]
    seq_len = diff.shape[2]
    out = {}

    for h in range(n_heads):
        valid_mask = torch.zeros((n_pos, seq_len), dtype=torch.bool, device=diff.device)
        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            if total_available <= 0:
                continue
            valid_mask[row_i, : min(total_available, seq_len)] = True

        if not valid_mask.any():
            raise ValueError("No valid entries for abs diff statistics.")

        signed_vals = diff[h][valid_mask]
        abs_vals = signed_vals.abs()
        coords = valid_mask.nonzero(as_tuple=False)

        k = min(int(topk), int(abs_vals.numel()))

        pos_vals, pos_idx = torch.topk(signed_vals, k=k, largest=True)
        neg_vals, neg_idx = torch.topk(signed_vals, k=k, largest=False)

        top_pos = []
        for rank in range(k):
            coord = coords[pos_idx[rank]]
            row_i = int(coord[0].item())
            key_pos = int(coord[1].item())
            v = float(pos_vals[rank].item())
            top_pos.append(
                {
                    "rank": rank + 1,
                    "query_pos": int(pos_list[row_i]),
                    "key_pos": key_pos,
                    "signed_diff": v,
                    "abs_diff": float(abs(v)),
                }
            )

        top_neg = []
        for rank in range(k):
            coord = coords[neg_idx[rank]]
            row_i = int(coord[0].item())
            key_pos = int(coord[1].item())
            v = float(neg_vals[rank].item())
            top_neg.append(
                {
                    "rank": rank + 1,
                    "query_pos": int(pos_list[row_i]),
                    "key_pos": key_pos,
                    "signed_diff": v,
                    "abs_diff": float(abs(v)),
                }
            )

        out[h] = {
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    return out


def print_signed_diff_topk_report(head_labels, signed_diff_topk):
    print("===== (optimal routing - attention routing) top positive/negative (Per Head) =====")
    print("Columns: head | rank | query_pos | key_pos | signed_diff | abs_diff")
    for i, h in enumerate(head_labels):
        print(f"-- head {h:>2d} positive top-k --")
        for item in signed_diff_topk[i]["top_positive"]:
            print(
                f"head {h:>2d} | "
                f"{item['rank']:>1d} | "
                f"{item['query_pos']:>4d} | "
                f"{item['key_pos']:>4d} | "
                f"{item['signed_diff']:+.6e} | "
                f"{item['abs_diff']:.6e}"
            )
        print(f"-- head {h:>2d} negative top-k --")
        for item in signed_diff_topk[i]["top_negative"]:
            print(
                f"head {h:>2d} | "
                f"{item['rank']:>1d} | "
                f"{item['query_pos']:>4d} | "
                f"{item['key_pos']:>4d} | "
                f"{item['signed_diff']:+.6e} | "
                f"{item['abs_diff']:.6e}"
            )


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
    probs = probs[:256,:256]
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
    x = x[:256,:256]
    return torch.sign(x) * torch.log10(1.0 + x.abs() / eps)


def build_diff_v_map(alpha_opt: torch.Tensor, alpha_base: torch.Tensor, v_abs: torch.Tensor):
    """Build delta-alpha weighted by |V| for visualization.

    alpha_*: [n_heads, n_pos, seq_len]
    v_abs:   [n_heads, seq_len]
    """
    if alpha_opt.shape != alpha_base.shape:
        raise ValueError(
            f"Shape mismatch: alpha_opt={tuple(alpha_opt.shape)} vs alpha_base={tuple(alpha_base.shape)}"
        )
    if v_abs.ndim != 2:
        raise ValueError(f"v_abs must have shape [n_heads, seq_len], got {tuple(v_abs.shape)}")
    if alpha_opt.shape[0] != v_abs.shape[0] or alpha_opt.shape[2] != v_abs.shape[1]:
        raise ValueError(
            "Shape mismatch for diff_v: "
            f"alpha shape={tuple(alpha_opt.shape)}, v_abs shape={tuple(v_abs.shape)}"
        )

    delta_alpha = alpha_opt - alpha_base
    return delta_alpha * v_abs.unsqueeze(1)


def summarize_sparsity_per_head(weights, pos_list, thresholds, mass_levels):
    """Summarize sparsity for [n_heads, n_pos, seq_len] weights on valid causal prefixes."""
    eps = 1e-12
    n_heads = weights.shape[0]
    out = {}

    thresholds = [float(x) for x in thresholds]
    mass_levels = sorted(float(x) for x in mass_levels)

    for h in range(n_heads):
        density_acc = {thr: [] for thr in thresholds}
        k_ratio_acc = {lvl: [] for lvl in mass_levels}
        entropy_acc = []

        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            if total_available <= 0:
                continue

            row = weights[h, row_i, :total_available].detach().float()

            for thr in thresholds:
                density_acc[thr].append((row > thr).float().mean().item())

            sorted_row = torch.sort(row, descending=True).values
            cumsum = torch.cumsum(sorted_row, dim=0)
            for lvl in mass_levels:
                idx = int(torch.searchsorted(cumsum, torch.tensor(lvl, device=row.device)).item())
                k = min(idx + 1, total_available)
                k_ratio_acc[lvl].append(k / float(total_available))

            entropy = -(row * torch.log(row.clamp_min(eps))).sum().item()
            entropy_norm = entropy / max(torch.log(torch.tensor(float(total_available))).item(), eps)
            entropy_acc.append(entropy_norm)

        head_stat = {
            "num_positions": len(entropy_acc),
            "entropy_norm_mean": float(torch.tensor(entropy_acc).mean().item()),
        }
        for thr in thresholds:
            key = f"density_gt_{thr:g}"
            head_stat[key] = float(torch.tensor(density_acc[thr]).mean().item())
        for lvl in mass_levels:
            key = f"k_ratio_for_mass_{lvl:g}"
            head_stat[key] = float(torch.tensor(k_ratio_acc[lvl]).mean().item())

        out[h] = head_stat

    return out


def summarize_sparsity_global(per_head_stats):
    keys = list(next(iter(per_head_stats.values())).keys())
    numeric_keys = [k for k in keys if k != "num_positions"]
    out = {"num_heads": len(per_head_stats)}
    for k in numeric_keys:
        vals = torch.tensor([float(v[k]) for v in per_head_stats.values()], dtype=torch.float32)
        out[f"{k}_mean"] = float(vals.mean().item())
        out[f"{k}_std"] = float(vals.std(unbiased=False).item())
    out["num_positions_mean"] = float(
        torch.tensor([float(v["num_positions"]) for v in per_head_stats.values()]).mean().item()
    )
    return out


def print_sparsity_report(head_labels, qk_raw_stats, qk_routing_stats, optimal_stats, mass_levels):
    first_lvl = float(sorted(mass_levels)[0])
    first_density_key = "density_gt_0.001"
    first_k_key = f"k_ratio_for_mass_{first_lvl:g}"

    print("===== Sparsity Check (Per Head) =====")
    print(
        "Columns: head | qk_raw dens>1e-3 | qk_route dens>1e-3 | opt dens>1e-3 | "
        f"qk_raw k@{first_lvl:g} | qk_route k@{first_lvl:g} | opt k@{first_lvl:g}"
    )

    for i, h in enumerate(head_labels):
        s_raw = qk_raw_stats[i]
        s_route = qk_routing_stats[i]
        s_opt = optimal_stats[i]
        print(
            f"head {h:>2d} | "
            f"{s_raw[first_density_key]:.4f} | "
            f"{s_route[first_density_key]:.4f} | "
            f"{s_opt[first_density_key]:.4f} | "
            f"{s_raw[first_k_key]:.4f} | "
            f"{s_route[first_k_key]:.4f} | "
            f"{s_opt[first_k_key]:.4f} |"
            f"{s_opt[first_k_key]-s_route[first_k_key]:.4f}"
        )


def compute_per_pos_sparsity_curves(weights, pos_list, mass_level=0.9):
    """Return per-position entropy_norm and k-ratio curves for [n_heads, n_pos, seq_len]."""
    eps = 1e-12
    n_heads = weights.shape[0]
    curves = {}

    for h in range(n_heads):
        entropy_curve = []
        k_ratio_curve = []

        for row_i, pos in enumerate(pos_list):
            total_available = int(pos) + 1
            row = weights[h, row_i, :total_available].detach().float()

            entropy = -(row * torch.log(row.clamp_min(eps))).sum().item()
            entropy_norm = entropy / max(torch.log(torch.tensor(float(total_available))).item(), eps)

            sorted_row = torch.sort(row, descending=True).values
            cumsum = torch.cumsum(sorted_row, dim=0)
            idx = int(torch.searchsorted(cumsum, torch.tensor(float(mass_level), device=row.device)).item())
            k = min(idx + 1, total_available)

            entropy_curve.append(float(entropy_norm))
            k_ratio_curve.append(float(k))

        curves[h] = {
            "entropy_norm": entropy_curve,
            "k_ratio": k_ratio_curve,
        }

    return curves


def plot_sparsity_curves_grid(
    head_labels,
    pos_list,
    qk_raw_curves,
    qk_routing_curves,
    optimal_curves,
    out_path,
    dpi,
    mass_level,
):
    n_heads = len(head_labels)
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 2)

    xs = list(pos_list)
    for i, head_label in enumerate(head_labels):
        c_raw = qk_raw_curves[i]
        c_route = qk_routing_curves[i]
        c_opt = optimal_curves[i]

        ax0 = axes[i, 0]
        # ax0.plot(xs, c_raw["entropy_norm"], label="qk_raw", linewidth=1.1)
        ax0.plot(xs, c_route["entropy_norm"], label="qk_routing", linewidth=1.2)
        ax0.plot(xs, c_opt["entropy_norm"], label="optimal", linewidth=1.2)
        ax0.set_ylim(0.0, 1.0)
        ax0.set_xlabel("Position")
        ax0.set_ylabel("Entropy (normalized)")
        ax0.set_title(f"Head {head_label}: per-pos entropy")
        ax0.grid(True, linestyle="--", alpha=0.35)
        ax0.legend()

        ax1 = axes[i, 1]
        # ax1.plot(xs, c_raw["k_ratio"], label="qk_raw", linewidth=1.1)
        ax1.plot(xs, c_route["k_ratio"], label="qk_routing", linewidth=1.2)
        ax1.plot(xs, c_opt["k_ratio"], label="optimal", linewidth=1.2)
        # ax1.set_ylim(0.0, 0.2)
        # ax1.set_yscale("log")
        ax1.set_xlabel("Position")
        ax1.set_ylabel(f"k-ratio for mass {mass_level:g}")
        ax1.set_title(f"Head {head_label}: per-pos k@{mass_level:g}")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_routing_grid(
    alpha_base,
    alpha_opt,
    diff_v,
    head_labels,
    out_path,
    dpi,
    alpha_viz,
    diff_log_eps,
):
    # alpha_*: [n_heads, n_pos, seq_len]
    n_heads = alpha_base.shape[0]
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 4, figsize=(20.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 4)

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
        diff_v_map = signed_log_map(diff_v[h], diff_log_eps)
        diff_v_lim = max(diff_v_map.abs().max().item(), 1e-8)

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

        im3 = axes[h, 3].imshow(
            _to_plot_array(diff_v_map),
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_v_lim,
            vmax=diff_v_lim,
        )
        axes[h, 3].set_title(f"Head {head_label}: diff_v=delta alpha*|V| signed-log (eps={diff_log_eps:g})")
        fig.colorbar(im3, ax=axes[h, 3], fraction=0.046)

        for col in range(4):
            axes[h, col].set_xlabel("Key position")
            axes[h, col].set_ylabel("Query position")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_overlap_grid(per_head, head_labels, out_path, dpi):
    n_heads = len(head_labels)
    fig_h = max(2.8 * n_heads, 6.0)
    fig, axes = plt.subplots(n_heads, 2, figsize=(14.0, fig_h), constrained_layout=True)

    if n_heads == 1:
        axes = axes.reshape(1, 2)

    for i, head_label in enumerate(head_labels):
        per_pos = per_head[i]
        positions = [x["pos"] for x in per_pos]
        overlap = [x["overlap_ratio"] for x in per_pos]
        jaccard = [x["jaccard"] for x in per_pos]
        mass1 = [x["opt_mass_on_base_topk"] for x in per_pos]
        mass2 = [x["base_mass_on_opt_topk"] for x in per_pos]

        axes[i, 0].plot(positions, overlap, label="top-k overlap ratio", linewidth=1.4)
        axes[i, 0].plot(positions, jaccard, label="jaccard", linewidth=1.2)
        axes[i, 0].set_ylim(0.0, 1.0)
        axes[i, 0].set_xlabel("Position")
        axes[i, 0].set_ylabel("Set overlap")
        axes[i, 0].set_title(f"Head {head_label}: overlap")
        axes[i, 0].grid(True, linestyle="--", alpha=0.35)
        axes[i, 0].legend()

        axes[i, 1].plot(positions, mass1, label="opt mass on base top-k", linewidth=1.4)
        axes[i, 1].plot(positions, mass2, label="base mass on opt top-k", linewidth=1.4)
        axes[i, 1].set_ylim(0.0, 1.0)
        axes[i, 1].set_xlabel("Position")
        axes[i, 1].set_ylabel("Cross mass")
        axes[i, 1].set_title(f"Head {head_label}: cross-mass")
        axes[i, 1].grid(True, linestyle="--", alpha=0.35)
        axes[i, 1].legend()

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

    qk_scores_all, qk_probs_all = get_attention_map_after_rope(
        ctx=layer_ctx,
        layer_idx=args.layer,
        causal=True,
        dtype=torch.float32,
        device=ctx.device,
    )
    qk_probs_selected = qk_probs_all[head_idx][:, pos_list, :]
    v_selected = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()
    v_abs = torch.norm(v_selected, p=2, dim=-1)

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

    signed_diff_top10_per_head = summarize_signed_diff_topk_per_head(
        alpha_opt=alpha_opt,
        alpha_base=alpha_baseline,
        pos_list=pos_list,
        topk=10,
    )
    diff_v = build_diff_v_map(
        alpha_opt=alpha_opt,
        alpha_base=alpha_baseline,
        v_abs=v_abs,
    )

    qk_raw_sparsity_per_head = summarize_sparsity_per_head(
        weights=qk_probs_selected,
        pos_list=pos_list,
        thresholds=args.sparsity_thresholds,
        mass_levels=args.sparsity_mass_levels,
    )
    qk_routing_sparsity_per_head = summarize_sparsity_per_head(
        weights=alpha_baseline,
        pos_list=pos_list,
        thresholds=args.sparsity_thresholds,
        mass_levels=args.sparsity_mass_levels,
    )
    optimal_sparsity_per_head = summarize_sparsity_per_head(
        weights=alpha_opt,
        pos_list=pos_list,
        thresholds=args.sparsity_thresholds,
        mass_levels=args.sparsity_mass_levels,
    )

    qk_raw_sparsity_global = summarize_sparsity_global(qk_raw_sparsity_per_head)
    qk_routing_sparsity_global = summarize_sparsity_global(qk_routing_sparsity_per_head)
    optimal_sparsity_global = summarize_sparsity_global(optimal_sparsity_per_head)

    per_pos_mass_level = 0.9
    qk_raw_per_pos_curves = compute_per_pos_sparsity_curves(
        weights=qk_probs_selected,
        pos_list=pos_list,
        mass_level=per_pos_mass_level,
    )
    qk_routing_per_pos_curves = compute_per_pos_sparsity_curves(
        weights=alpha_baseline,
        pos_list=pos_list,
        mass_level=per_pos_mass_level,
    )
    optimal_per_pos_curves = compute_per_pos_sparsity_curves(
        weights=alpha_opt,
        pos_list=pos_list,
        mass_level=per_pos_mass_level,
    )

    mat_path = os.path.join(output_dir, "routing_grid.png")
    overlap_curve_path = os.path.join(output_dir, "topk_overlap_curve.png")
    sparsity_curve_path = os.path.join(output_dir, "sparsity_curves_per_pos.png")
    stats_path = os.path.join(output_dir, "overlap_stats.pt")

    plot_routing_grid(
        alpha_base=alpha_baseline,
        alpha_opt=alpha_opt,
        diff_v=diff_v,
        head_labels=head_idx,
        out_path=mat_path,
        dpi=args.dpi,
        alpha_viz=args.alpha_viz,
        diff_log_eps=args.diff_log_eps,
    )

    plot_overlap_grid(
        per_head=per_head,
        head_labels=head_idx,
        out_path=overlap_curve_path,
        dpi=args.dpi,
    )

    plot_sparsity_curves_grid(
        head_labels=head_idx,
        pos_list=pos_list,
        qk_raw_curves=qk_raw_per_pos_curves,
        qk_routing_curves=qk_routing_per_pos_curves,
        optimal_curves=optimal_per_pos_curves,
        out_path=sparsity_curve_path,
        dpi=args.dpi,
        mass_level=per_pos_mass_level,
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
            "signed_diff_top10_per_head": signed_diff_top10_per_head,
            "sparsity": {
                "thresholds": args.sparsity_thresholds,
                "mass_levels": args.sparsity_mass_levels,
                "per_pos_mass_level": per_pos_mass_level,
                "qk_raw_per_head": qk_raw_sparsity_per_head,
                "qk_routing_per_head": qk_routing_sparsity_per_head,
                "optimal_per_head": optimal_sparsity_per_head,
                "qk_raw_global": qk_raw_sparsity_global,
                "qk_routing_global": qk_routing_sparsity_global,
                "optimal_global": optimal_sparsity_global,
                "qk_raw_per_pos_curves": qk_raw_per_pos_curves,
                "qk_routing_per_pos_curves": qk_routing_per_pos_curves,
                "optimal_per_pos_curves": optimal_per_pos_curves,
            },
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
    print(
        "global density_gt_1e-3: "
        f"qk_raw={qk_raw_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}, "
        f"qk_routing={qk_routing_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}, "
        f"optimal={optimal_sparsity_global.get('density_gt_0.001_mean', float('nan')):.6f}"
    )
    print(
        f"global k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}: "
        f"qk_raw={qk_raw_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}, "
        f"qk_routing={qk_routing_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}, "
        f"optimal={optimal_sparsity_global.get(f'k_ratio_for_mass_{args.sparsity_mass_levels[0]:g}_mean', float('nan')):.6f}"
    )
    print_sparsity_report(
        head_labels=head_idx,
        qk_raw_stats=qk_raw_sparsity_per_head,
        qk_routing_stats=qk_routing_sparsity_per_head,
        optimal_stats=optimal_sparsity_per_head,
        mass_levels=args.sparsity_mass_levels,
    )
    print_signed_diff_topk_report(
        head_labels=head_idx,
        signed_diff_topk=signed_diff_top10_per_head,
    )
    print(f"saved matrix plot: {mat_path}")
    print(f"saved overlap curve: {overlap_curve_path}")
    print(f"saved sparsity curve: {sparsity_curve_path}")
    print(f"saved stats: {stats_path}")


if __name__ == "__main__":
    main()