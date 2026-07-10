"""
Visualize q dot delta K_i / sqrt(d) inside sampled routing clusters.

For each sampled query/head/cluster, this script plots:
    qk_i = q dot K_i / sqrt(d)
    delta K_i = K_i - avg_{j in C}(K_j)
    q_delta_k_i = q dot delta K_i / sqrt(d)

For each sampled query/head, this script also plots the distribution of
softmax-normalized cluster attention masses:
    cluster_mass_C = sum_{i in C} softmax(q dot K / sqrt(d))_i

The representative K_rep is the kept token that the cluster belongs to under
gen_mask_h2o_with_belong_all.
"""

import argparse
import math
import os
import random
import warnings

import matplotlib.pyplot as plt
import torch

from ..attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong, get_qk_logits
from ..experiment_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
    resolve_output_dir,
    validate_common_args,
)
from ..config import set_seed, str_to_torch_dtype
from ..online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ..runtime import load_context
from ..sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot exp(qk_i) distributions inside sampled clusters."
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
        "--query-positions",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Absolute query positions to plot. Default: randomly sample from "
            "--sample-query-start/--sample-query-end inside --pos-start/--pos-end."
        ),
    )
    parser.add_argument("--sample-heads", type=int, default=4)
    parser.add_argument("--sample-queries", type=int, default=4)
    parser.add_argument(
        "--sample-query-start",
        type=int,
        default=512,
        help="Inclusive lower bound for randomly sampled query positions used in plots.",
    )
    parser.add_argument(
        "--sample-query-end",
        type=int,
        default=1024,
        help="Exclusive upper bound for randomly sampled query positions used in plots.",
    )
    parser.add_argument("--clusters-per-query", type=int, default=6)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="Number of histogram bins for exp(qk) plots. Larger values use smaller bin spacing.",
    )
    parser.add_argument(
        "--cluster-sample-mode",
        choices=["largest", "random"],
        default="largest",
        help="Select largest clusters or randomly sample eligible clusters.",
    )
    parser.add_argument("--cluster-seed", type=int, default=42)
    return parser.parse_args()


def _stats(vals):
    vals = vals.detach().float().cpu()
    if vals.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "frac_abs_lt_0.1": float("nan"),
            "frac_abs_lt_0.5": float("nan"),
            "frac_abs_lt_1": float("nan"),
            "frac_abs_lt_2": float("nan"),
        }

    abs_vals = vals.abs()
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "min": float(vals.min().item()),
        "p05": _percentile(vals, 0.05),
        "median": float(vals.median().item()),
        "p95": _percentile(vals, 0.95),
        "max": float(vals.max().item()),
        "frac_abs_lt_0.1": float((abs_vals < 0.1).float().mean().item()),
        "frac_abs_lt_0.5": float((abs_vals < 0.5).float().mean().item()),
        "frac_abs_lt_1": float((abs_vals < 1.0).float().mean().item()),
        "frac_abs_lt_2": float((abs_vals < 2.0).float().mean().item()),
    }


def _percentile(vals, q):
    vals = vals.reshape(-1)
    if vals.numel() == 0:
        return float("nan")
    k = int(math.ceil(float(q) * vals.numel()))
    k = min(max(k, 1), vals.numel())
    return float(vals.kthvalue(k).values.item())


def _choose_evenly(xs, n):
    if len(xs) <= n:
        return list(xs)
    if n <= 1:
        return [xs[len(xs) // 2]]
    idx = torch.linspace(0, len(xs) - 1, steps=n).round().long().tolist()
    return [xs[i] for i in idx]


def _sample_positions(xs, n, rng):
    if len(xs) <= n:
        return list(xs)
    return sorted(rng.sample(list(xs), n))


def _select_clusters(row_root, route_mask_row, total_available, min_cluster_size, max_clusters, mode, rng):
    kept = (~torch.isneginf(route_mask_row[:total_available])).nonzero(as_tuple=False).squeeze(-1)
    candidates = []
    for root in kept.tolist():
        members = (row_root[:total_available] == root).nonzero(as_tuple=False).squeeze(-1)
        if members.numel() >= min_cluster_size:
            candidates.append((root, members))

    if mode == "largest":
        candidates.sort(key=lambda item: int(item[1].numel()), reverse=True)
        return candidates[:max_clusters]

    rng.shuffle(candidates)
    return candidates[:max_clusters]


def _all_clusters(row_root, route_mask_row, total_available):
    kept = (~torch.isneginf(route_mask_row[:total_available])).nonzero(as_tuple=False).squeeze(-1)
    clusters = []
    for root in kept.tolist():
        members = (row_root[:total_available] == root).nonzero(as_tuple=False).squeeze(-1)
        if members.numel() > 0:
            clusters.append((root, members))
    return clusters


def _plot_cluster_page(out_path, clusters, title, dpi, hist_bins):
    n = len(clusters)
    if n == 0:
        return False

    fig_h = max(2.6 * n, 3.0)
    fig, axes = plt.subplots(n, 1, figsize=(6.8, fig_h), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for row, cluster in enumerate(clusters):
        q_delta_k = cluster["q_delta_k"].detach().float().cpu()
        q_delta_k_rep = float(cluster["q_delta_k_rep"])

        ax = axes[row]
        bins = max(1, int(hist_bins))
        bin_width = (
            float((q_delta_k.max() - q_delta_k.min()).item()) / bins
            if q_delta_k.numel() > 1
            else 0.0
        )
        ax.hist(q_delta_k.numpy(), bins=bins, alpha=0.82, color="#3b82f6")
        ax.axvline(
            q_delta_k_rep,
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label="rep q delta K",
        )
        ax.set_title(
            f"h{cluster['head']} q{cluster['pos']} root{cluster['root']} "
            f"size={cluster['size']} | q delta K / sqrt(d) | bin_width={bin_width:.3g}"
        )
        ax.set_xlabel("q dot (K_i - avg K_cluster) / sqrt(d)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_cluster_mass_page(out_path, query_rows, title, dpi, hist_bins):
    n = len(query_rows)
    if n == 0:
        return False

    fig_h = max(2.4 * n, 3.0)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, fig_h), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for row_idx, row in enumerate(query_rows):
        masses = row["cluster_masses"].detach().float().cpu()
        ax = axes[row_idx]
        bins = max(1, min(int(hist_bins), max(1, masses.numel())))
        bin_width = float((masses.max() - masses.min()).item()) / bins if masses.numel() > 1 else 0.0
        ax.hist(masses.numpy(), bins=bins, alpha=0.82, color="#10b981")
        ax.axvline(float(masses.mean().item()), color="#111827", linewidth=1.5, linestyle="--", label="mean")
        ax.set_title(
            f"h{row['head']} q{row['pos']} | clusters={row['num_clusters']} "
            f"| total={row['total_mass']:.3g} | bin_width={bin_width:.3g}"
        )
        ax.set_xlabel("sum_{i in C} softmax(q dot K / sqrt(d))_i")
        ax.set_ylabel("cluster count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _save_cluster_stats(out_path, rows):
    stat_names = [
        "mean",
        "std",
        "min",
        "p05",
        "median",
        "p95",
        "max",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "head\tquery_pos\troot\tsize\tq_delta_k_rep\tmetric\t"
            + "\t".join(stat_names)
            + "\n"
        )
        for row in rows:
            vals = [row[f"q_delta_k_{name}"] for name in stat_names]
            f.write(
                f"{row['head']}\t{row['pos']}\t{row['root']}\t{row['size']}\t"
                f"{row['q_delta_k_rep']:.6f}\tq_delta_k\t"
                + "\t".join(f"{v:.6f}" for v in vals)
                + "\n"
            )


def _save_cluster_mass_stats(out_path, rows):
    stat_names = [
        "mean",
        "std",
        "min",
        "p05",
        "median",
        "p95",
        "max",
        "frac_abs_lt_0.1",
        "frac_abs_lt_0.5",
        "frac_abs_lt_1",
        "frac_abs_lt_2",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "head\tquery_pos\tnum_clusters\ttotal_softmax_mass\tmetric\t"
            + "\t".join(stat_names)
            + "\n"
        )
        for row in rows:
            vals = [row[f"cluster_mass_{name}"] for name in stat_names]
            f.write(
                f"{row['head']}\t{row['pos']}\t{row['num_clusters']}\t"
                f"{row['total_mass']:.6f}\tcluster_softmax_mass\t"
                + "\t".join(f"{v:.6f}" for v in vals)
                + "\n"
            )


def _save_cluster_mass_values(out_path, rows):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("head\tquery_pos\troot\tsize\tcluster_softmax_mass\n")
        for row in rows:
            roots = row["roots"].tolist()
            sizes = row["sizes"].tolist()
            masses = row["cluster_masses"].tolist()
            for root, size, mass in zip(roots, sizes, masses):
                f.write(
                    f"{row['head']}\t{row['pos']}\t{int(root)}\t{int(size)}\t"
                    f"{float(mass):.6f}\n"
                )


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


def collect_and_plot_clusters(
    *,
    out_dir,
    qk_logits,
    route_mask,
    belong_root,
    pos_list,
    head_labels,
    args,
):
    rng = random.Random(args.cluster_seed)
    os.makedirs(out_dir, exist_ok=True)

    selected_heads = head_labels[:]
    if len(selected_heads) > args.sample_heads:
        selected_heads = _choose_evenly(selected_heads, args.sample_heads)

    if args.query_positions is not None:
        selected_pos = [pos for pos in args.query_positions if pos in set(pos_list)]
        missing = sorted(set(args.query_positions) - set(selected_pos))
        if missing:
            warnings.warn(f"Ignore query positions outside selected pos range: {missing}", stacklevel=2)
    else:
        sample_end = args.sample_query_end if args.sample_query_end is not None else max(pos_list) + 1
        sample_candidates = [
            pos for pos in pos_list if args.sample_query_start <= pos < min(sample_end, args.seq_len)
        ]
        if len(sample_candidates) == 0:
            raise ValueError(
                "No query positions available for plotting after applying "
                f"--sample-query-start={args.sample_query_start}, "
                f"--sample-query-end={args.sample_query_end}, and --pos-start/--pos-end."
            )
        selected_pos = _sample_positions(sample_candidates, args.sample_queries, rng)

    pos_to_i = {pos: i for i, pos in enumerate(pos_list)}
    head_to_h = {head: i for i, head in enumerate(head_labels)}

    cluster_rows = []
    cluster_mass_rows = []
    cluster_mass_tensors = []
    plot_paths = []
    tensor_clusters = []

    for head in selected_heads:
        h = head_to_h[head]
        for pos in selected_pos:
            i = pos_to_i[pos]
            total_available = pos + 1
            all_clusters = _all_clusters(
                row_root=belong_root[h, i],
                route_mask_row=route_mask[h, i],
                total_available=total_available,
            )
            mass_values = []
            mass_roots = []
            mass_sizes = []
            row_logits = qk_logits[h, i, :total_available].float()
            row_weights = torch.softmax(row_logits, dim=0)
            for root, members in all_clusters:
                mass_values.append(row_weights[members].sum())
                mass_roots.append(int(root))
                mass_sizes.append(int(members.numel()))
            if mass_values:
                cluster_masses = torch.stack(mass_values).detach().cpu()
                mass_stats = _stats(cluster_masses)
                mass_row = {
                    "head": int(head),
                    "pos": int(pos),
                    "num_clusters": int(cluster_masses.numel()),
                    "total_mass": float(cluster_masses.sum().item()),
                }
                for name, value in mass_stats.items():
                    mass_row[f"cluster_mass_{name}"] = value
                cluster_mass_rows.append(mass_row)
                cluster_mass_tensors.append(
                    {
                        "head": int(head),
                        "pos": int(pos),
                        "roots": torch.tensor(mass_roots, dtype=torch.long),
                        "sizes": torch.tensor(mass_sizes, dtype=torch.long),
                        "cluster_masses": cluster_masses,
                        "total_mass": float(cluster_masses.sum().item()),
                        "num_clusters": int(cluster_masses.numel()),
                    }
                )

            clusters = _select_clusters(
                row_root=belong_root[h, i],
                route_mask_row=route_mask[h, i],
                total_available=total_available,
                min_cluster_size=args.min_cluster_size,
                max_clusters=args.clusters_per_query,
                mode=args.cluster_sample_mode,
                rng=rng,
            )

            plot_clusters = []
            for root, members in clusters:
                qk = qk_logits[h, i, members].float()
                qk_rep = qk_logits[h, i, root].float()
                qk_cluster_mean = qk.mean()
                q_delta_k = qk - qk_cluster_mean
                q_delta_k_rep = qk_rep - qk_cluster_mean
                q_delta_k_stats = _stats(q_delta_k)

                row = {
                    "head": int(head),
                    "pos": int(pos),
                    "root": int(root),
                    "size": int(members.numel()),
                    "q_delta_k_rep": float(q_delta_k_rep.item()),
                }
                for name, value in q_delta_k_stats.items():
                    row[f"q_delta_k_{name}"] = value
                cluster_rows.append(row)

                cluster = {
                    "head": int(head),
                    "pos": int(pos),
                    "root": int(root),
                    "size": int(members.numel()),
                    "members": members.detach().cpu(),
                    "q_delta_k": q_delta_k.detach().cpu(),
                    "q_delta_k_rep": float(q_delta_k_rep.item()),
                }
                plot_clusters.append(cluster)
                tensor_clusters.append(cluster)

            if len(plot_clusters) > 0:
                plot_path = os.path.join(out_dir, f"cluster_h{head}_q{pos}.png")
                ok = _plot_cluster_page(
                    plot_path,
                    plot_clusters,
                    title=f"Cluster q delta K / sqrt(d) distributions | head={head}, query={pos}",
                    dpi=args.plot_dpi,
                    hist_bins=args.hist_bins,
                )
                if ok:
                    plot_paths.append(plot_path)

    stats_path = os.path.join(out_dir, "cluster_q_delta_k_stats.tsv")
    _save_cluster_stats(stats_path, cluster_rows)

    mass_stats_path = os.path.join(out_dir, "cluster_softmax_mass_stats.tsv")
    _save_cluster_mass_stats(mass_stats_path, cluster_mass_rows)

    mass_values_path = os.path.join(out_dir, "cluster_softmax_mass_values.tsv")
    _save_cluster_mass_values(mass_values_path, cluster_mass_tensors)

    mass_plot_path = os.path.join(out_dir, "cluster_softmax_mass_distribution.png")
    if _plot_cluster_mass_page(
        mass_plot_path,
        cluster_mass_tensors,
        title="Cluster softmax attention mass distributions",
        dpi=args.plot_dpi,
        hist_bins=args.hist_bins,
    ):
        plot_paths.append(mass_plot_path)

    tensor_path = os.path.join(out_dir, "cluster_q_delta_k_samples.pt")
    torch.save(
        {
            "head_labels": head_labels,
            "selected_heads": selected_heads,
            "selected_pos": selected_pos,
            "clusters": tensor_clusters,
            "cluster_softmax_masses": cluster_mass_tensors,
            "config": vars(args),
        },
        tensor_path,
    )

    print("===== Cluster q dot delta K / sqrt(d) Summary =====")
    print(f"sampled_heads={selected_heads}")
    print(f"sampled_query_positions={selected_pos}")
    print(f"sampled_clusters={len(cluster_rows)}")
    print(f"Saved cluster stats to: {stats_path}")
    print(f"Saved cluster softmax mass stats to: {mass_stats_path}")
    print(f"Saved cluster softmax mass values to: {mass_values_path}")
    print(f"Saved cluster samples to: {tensor_path}")
    for path in plot_paths:
        print(f"Saved cluster plot to: {path}")

    return {
        "stats_path": stats_path,
        "mass_stats_path": mass_stats_path,
        "mass_values_path": mass_values_path,
        "mass_plot_path": mass_plot_path,
        "tensor_path": tensor_path,
        "plot_paths": plot_paths,
    }


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
        compare_tag="compare_error_clusterplot",
        include_loss_type=True,
    )

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
        prefix_patches = _align_prefix_patches_to_pos_list(prefix_patches, pos_list, args.seq_len)
    else:
        prefix_patches = build_baseline_prefix_patches(
            ctx=ctx,
            args=args,
            target_layer=args.layer,
            pos_list=pos_list,
            model_inputs=model_inputs,
            build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask_h2o_with_belong_all(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
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
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong, _count = gen_mask_h2o_with_belong_all(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        merge_metric=args.merge_metric,
    )
    belong_root = canonicalize_belong(belong, pos_list)

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=args.layer,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    result = collect_and_plot_clusters(
        out_dir=output_dir,
        qk_logits=qk_logits,
        route_mask=route_mask,
        belong_root=belong_root,
        pos_list=pos_list,
        head_labels=head_idx,
        args=args,
    )

    summary_path = os.path.join(output_dir, "compare_error_clusterplot_summary.pt")
    torch.save(
        {
            "config": vars(args),
            "layer": int(args.layer),
            "heads": head_idx,
            "pos_list": pos_list,
            "outputs": result,
        },
        summary_path,
    )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
