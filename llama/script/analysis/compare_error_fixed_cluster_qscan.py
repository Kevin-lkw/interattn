"""
Scan signed q dot delta K scores for clusters fixed at one reference query.

For a reference query position, choose several routing clusters.  For each
query position in a scan range, measure

    q_delta_k_i = q dot (K_i - avg_{j in C} K_j) / sqrt(d)

over the fixed cluster members C.
"""

import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import torch

from .attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong, get_qk_logits
from .compare_error_clusterplot import _percentile
from .compare_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
    resolve_output_dir,
    validate_common_args,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan q_delta_k scores for clusters fixed at one reference position."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
    )
    parser.add_argument(
        "--merge-metric",
        choices=["k", "v"],
        default="k",
        help="Cluster merge target metric.",
    )
    parser.add_argument(
        "--cluster-pos",
        type=int,
        default=512,
        help="Reference query position whose routing clusters are reused.",
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=512,
        help="Inclusive start position for scanned queries.",
    )
    parser.add_argument(
        "--query-end",
        type=int,
        default=612,
        help="Exclusive end position for scanned queries.",
    )
    parser.add_argument(
        "--sample-heads",
        type=int,
        default=4,
        help="When --head/--heads is not set, choose this many heads evenly across all heads.",
    )
    parser.add_argument("--clusters-per-head", type=int, default=12)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument(
        "--cluster-sample-mode",
        choices=["largest", "random"],
        default="largest",
        help="Select largest clusters or randomly sample eligible clusters at --cluster-pos.",
    )
    parser.add_argument("--cluster-seed", type=int, default=42)
    return parser.parse_args()


def _align_prefix_patches_to_pos_list(prefix_patches, pos_list, seq_len):
    pos_idx = torch.tensor(pos_list, dtype=torch.long)
    aligned = {}
    for layer_idx, patch_hidden in prefix_patches.items():
        if patch_hidden.shape[0] == len(pos_list):
            aligned[layer_idx] = patch_hidden
        elif patch_hidden.shape[0] == seq_len:
            aligned[layer_idx] = patch_hidden.index_select(0, pos_idx.to(patch_hidden.device))
        else:
            raise ValueError(
                f"prefix patch for layer {layer_idx} has {patch_hidden.shape[0]} positions; "
                f"expected len(pos_list)={len(pos_list)} or seq_len={seq_len}"
            )
    return aligned


def _stats(vals):
    vals = vals.detach().float().cpu()
    if vals.numel() == 0:
        return {
            "rep": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "std": float(vals.std(unbiased=False).item()),
        "min": float(vals.min().item()),
        "p05": _percentile(vals, 0.05),
        "median": float(vals.median().item()),
        "p95": _percentile(vals, 0.95),
        "max": float(vals.max().item()),
    }


def _select_fixed_clusters(row_root, route_mask_row, total_available, min_cluster_size, max_clusters, mode, rng):
    def collect_from_roots(roots):
        clusters = []
        seen = set()
        for root in roots:
            root = int(root)
            if root in seen:
                continue
            seen.add(root)
            members = (row_root[:total_available] == root).nonzero(as_tuple=False).squeeze(-1)
            if members.numel() >= min_cluster_size:
                clusters.append((root, members))
        return clusters

    kept = (~torch.isneginf(route_mask_row[:total_available])).nonzero(as_tuple=False).squeeze(-1)
    clusters = collect_from_roots(kept.tolist())

    if len(clusters) == 0:
        roots = torch.unique(row_root[:total_available]).tolist()
        clusters = collect_from_roots(roots)

    if mode == "largest":
        clusters.sort(key=lambda item: int(item[1].numel()), reverse=True)
        return clusters[:max_clusters]

    rng.shuffle(clusters)
    return clusters[:max_clusters]


def _choose_evenly(xs, n):
    if len(xs) <= n:
        return list(xs)
    if n <= 1:
        return [xs[len(xs) // 2]]
    idx = torch.linspace(0, len(xs) - 1, steps=n).round().long().tolist()
    return [xs[i] for i in idx]


def _save_scan_tsv(out_path, rows):
    stat_names = ["rep", "std", "min", "p05", "median", "p95", "max"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "head\tcluster_pos\tquery_pos\tcluster_rank\troot\tsize\t"
            + "\t".join(stat_names)
            + "\n"
        )
        for row in rows:
            vals = [row[name] for name in stat_names]
            f.write(
                f"{row['head']}\t{row['cluster_pos']}\t{row['query_pos']}\t"
                f"{row['cluster_rank']}\t{row['root']}\t{row['size']}\t"
                + "\t".join(f"{v:.6f}" for v in vals)
                + "\n"
            )


def _plot_cluster_scan(out_path, cluster_scans, title, dpi):
    if len(cluster_scans) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 3.2), constrained_layout=True)
        ax.text(
            0.5,
            0.5,
            "No eligible fixed clusters",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(title)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return True

    fig_h = max(2.5 * len(cluster_scans), 3.2)
    fig, axes = plt.subplots(len(cluster_scans), 1, figsize=(8.2, fig_h), constrained_layout=True)
    if len(cluster_scans) == 1:
        axes = [axes]

    for ax, scan in zip(axes, cluster_scans):
        x = scan["query_positions"]
        p05 = scan["p05"]
        p95 = scan["p95"]
        median = scan["median"]
        rep = scan["rep"]
        ax.fill_between(x, p05, p95, color="#93c5fd", alpha=0.35, label="p05-p95")
        ax.plot(x, median, color="#2563eb", linewidth=1.4, label="median")
        ax.plot(x, rep, color="#111827", linewidth=1.1, linestyle="--", label="rep")
        ax.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.7)
        ax.set_title(
            f"h{scan['head']} ref_q{scan['cluster_pos']} cluster#{scan['cluster_rank']} "
            f"root{scan['root']} size={scan['size']}"
        )
        ax.set_xlabel("query position")
        ax.set_ylabel("q delta K")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def collect_fixed_cluster_qscan(
    *,
    out_dir,
    qk_logits,
    route_mask,
    belong_root,
    pos_list,
    head_labels,
    args,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.cluster_seed)

    if args.cluster_pos not in pos_list:
        raise ValueError(f"--cluster-pos={args.cluster_pos} is not in computed pos_list")
    query_positions = list(range(args.query_start, args.query_end))
    missing = [pos for pos in query_positions if pos not in pos_list]
    if missing:
        raise ValueError(f"Query positions not in computed pos_list: {missing[:8]}")
    if query_positions and min(query_positions) < args.cluster_pos:
        raise ValueError("--query-start must be >= --cluster-pos so fixed cluster keys are visible")

    pos_to_i = {pos: i for i, pos in enumerate(pos_list)}
    cluster_i = pos_to_i[args.cluster_pos]
    query_i = [pos_to_i[pos] for pos in query_positions]

    rows = []
    cluster_scans = []
    tensor_scans = []

    for head_ord, head in enumerate(head_labels):
        clusters = _select_fixed_clusters(
            row_root=belong_root[head_ord, cluster_i],
            route_mask_row=route_mask[head_ord, cluster_i],
            total_available=args.cluster_pos + 1,
            min_cluster_size=args.min_cluster_size,
            max_clusters=args.clusters_per_head,
            mode=args.cluster_sample_mode,
            rng=rng,
        )
        if len(clusters) == 0:
            warnings.warn(f"No eligible clusters at head={head}, pos={args.cluster_pos}", stacklevel=2)
            continue

        for cluster_rank, (root, members) in enumerate(clusters):
            qk_cluster = qk_logits[head_ord, query_i][:, members].float()
            qk_mean = qk_cluster.mean(dim=1, keepdim=True)
            q_delta_k = qk_cluster - qk_mean
            rep_idx = (members == root).nonzero(as_tuple=False).squeeze(-1)
            if rep_idx.numel() != 1:
                raise ValueError(f"root {root} is not a unique member for head={head}")
            rep = q_delta_k[:, int(rep_idx.item())]

            p05 = []
            p95 = []
            median = []
            std = []
            min_vals = []
            max_vals = []
            for row_idx, pos in enumerate(query_positions):
                vals = q_delta_k[row_idx]
                s = _stats(vals)
                s["rep"] = float(rep[row_idx].item())
                rows.append(
                    {
                        "head": int(head),
                        "cluster_pos": int(args.cluster_pos),
                        "query_pos": int(pos),
                        "cluster_rank": int(cluster_rank),
                        "root": int(root),
                        "size": int(members.numel()),
                        **s,
                    }
                )
                p05.append(s["p05"])
                p95.append(s["p95"])
                median.append(s["median"])
                std.append(s["std"])
                min_vals.append(s["min"])
                max_vals.append(s["max"])

            scan = {
                "head": int(head),
                "cluster_pos": int(args.cluster_pos),
                "cluster_rank": int(cluster_rank),
                "root": int(root),
                "size": int(members.numel()),
                "query_positions": query_positions,
                "rep": rep.detach().cpu().tolist(),
                "std": std,
                "min": min_vals,
                "p05": p05,
                "median": median,
                "p95": p95,
                "max": max_vals,
            }
            cluster_scans.append(scan)
            tensor_scans.append(
                {
                    "head": int(head),
                    "cluster_pos": int(args.cluster_pos),
                    "cluster_rank": int(cluster_rank),
                    "root": int(root),
                    "members": members.detach().cpu(),
                    "query_positions": torch.tensor(query_positions, dtype=torch.long),
                    "q_delta_k": q_delta_k.detach().cpu(),
                    "rep": rep.detach().cpu(),
                }
            )

    stats_path = os.path.join(out_dir, "fixed_cluster_qscan_stats.tsv")
    _save_scan_tsv(stats_path, rows)

    plot_path = os.path.join(out_dir, "fixed_cluster_qscan.png")
    _plot_cluster_scan(
        plot_path,
        cluster_scans,
        title=(
            f"Fixed clusters from q={args.cluster_pos}; "
            f"scan q={args.query_start}..{args.query_end - 1}"
        ),
        dpi=args.plot_dpi,
    )

    tensor_path = os.path.join(out_dir, "fixed_cluster_qscan_samples.pt")
    torch.save(
        {
            "head_labels": head_labels,
            "cluster_pos": int(args.cluster_pos),
            "query_positions": query_positions,
            "clusters": tensor_scans,
            "config": vars(args),
        },
        tensor_path,
    )

    print("===== Fixed cluster q_delta_k scan =====")
    print(f"cluster_pos={args.cluster_pos} query_range=[{args.query_start}, {args.query_end})")
    print(f"heads={head_labels} clusters={len(tensor_scans)} rows={len(rows)}")
    print(f"Saved stats to: {stats_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved samples to: {tensor_path}")

    return {
        "stats_path": stats_path,
        "plot_path": plot_path,
        "tensor_path": tensor_path,
    }


def _build_routing_pos_list(args):
    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    if args.pos_start > args.cluster_pos:
        raise ValueError("--pos-start must be <= --cluster-pos for normal cache construction")
    if pos_end < args.query_end:
        raise ValueError("--pos-end must be >= --query-end for qscan output positions")

    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty routing pos_list after applying --pos-start/--pos-end")
    return pos_list


def main():
    set_seed(42)
    args = parse_args()

    if args.query_end <= args.query_start:
        raise ValueError("--query-end must be larger than --query-start")
    if args.cluster_pos < 0 or args.cluster_pos >= args.seq_len:
        raise ValueError("--cluster-pos must be inside [0, --seq-len)")
    if args.query_end > args.seq_len:
        raise ValueError("--query-end must be <= --seq-len")

    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_common_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )

    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    if args.head is None and args.heads is None:
        head_idx = _choose_evenly(head_idx, args.sample_heads)
    routing_pos_list = _build_routing_pos_list(args)
    print(
        f"Build routing clusters with pos_list=[{routing_pos_list[0]}, {routing_pos_list[-1]}] "
        f"({len(routing_pos_list)} positions); output qscan=[{args.query_start}, {args.query_end})"
    )
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(
        args=args,
        head_idx=head_idx,
        compare_tag="compare_error_fixed_cluster_qscan",
        include_loss_type=True,
    )

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
        prefix_patches = _align_prefix_patches_to_pos_list(prefix_patches, routing_pos_list, args.seq_len)
    else:
        prefix_patches = build_baseline_prefix_patches(
            ctx=ctx,
            args=args,
            target_layer=args.layer,
            pos_list=routing_pos_list,
            model_inputs=model_inputs,
            build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask_h2o_with_belong_all(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=routing_pos_list,
                head_idx=hi,
                budget=args.budget,
                seq_len=args.seq_len,
                adaptive_budget=args.adaptive_budget,
                merge_metric=args.merge_metric,
            )[0],
        )

    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=routing_pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong, _count = gen_mask_h2o_with_belong_all(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=routing_pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        merge_metric=args.merge_metric,
    )
    belong_root = canonicalize_belong(belong, routing_pos_list)

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=routing_pos_list,
        device=ctx.device,
    )

    result = collect_fixed_cluster_qscan(
        out_dir=output_dir,
        qk_logits=qk_logits,
        route_mask=route_mask,
        belong_root=belong_root,
        pos_list=routing_pos_list,
        head_labels=head_idx,
        args=args,
    )

    summary_path = os.path.join(output_dir, "compare_error_fixed_cluster_qscan_summary.pt")
    torch.save(
        {
            "config": vars(args),
            "layer": int(args.layer),
            "heads": head_idx,
            "pos_list": routing_pos_list,
            "outputs": result,
        },
        summary_path,
    )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
