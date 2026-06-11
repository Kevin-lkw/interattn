"""
Inspect cluster-level conditioning terms for H2O count-all routing.

For each selected query q and each cluster C at that query:

    s_C     = q dot mean(K_C) / sqrt(d)
    Z_C     = |C| exp(s_C)
    p_C     = Z_C / sum_C' Z_C'  (mean-K approximation)
    p_full_C= sum_i in C exp(q dot K_i / sqrt(d)) / full softmax denominator
    delta_C = max_i in C |q dot (K_i - mean(K_C)) / sqrt(d)|
    B_C     = max_i in C ||V_i||

The plotted condition term uses p_C as p_hat_C and reports the bound contribution:

    p_C * (2B * (cosh(delta_C) - 1) / sum_C' p_C' cosh(delta_C')
           + 2B_C * tanh(delta_C / 2))
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch

from .attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong
from .compare_error_fixed_cluster_qscan import (
    _align_prefix_patches_to_pos_list,
    _choose_evenly,
)
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
        description="Plot and tabulate cluster-level p, delta, and conditioning terms."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["h2o"],
        default_strategy="h2o",
        include_loss_type=True,
        include_plot_dpi=True,
        prefix_mode_default="full_attention",
        prefix_mode_choices=("full_attention", "optimal_saved", "baseline_rebuild"),
        prefix_mode_help=(
            "How to prepare layers before target layer. "
            "full_attention: keep previous layers unpatched; "
            "optimal_saved: load saved optimal patch_hidden for layers < target; "
            "baseline_rebuild: rebuild baseline patches online for layers < target."
        ),
    )
    parser.add_argument(
        "--merge-metric",
        choices=["k", "v"],
        default="k",
        help="Cluster merge target metric.",
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=256,
        help="Inclusive start position for query selection.",
    )
    parser.add_argument(
        "--query-end",
        type=int,
        default=None,
        help="Exclusive end position for query selection. Defaults to --pos-end/--seq-len.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=16,
        help="Number of query positions to plot per head.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        default=None,
        help="Explicit query positions. Overrides --query-start/--query-end/--num-queries.",
    )
    parser.add_argument(
        "--sample-heads",
        type=int,
        default=4,
        help="When --head/--heads is not set, choose this many heads evenly across all heads.",
    )
    parser.add_argument(
        "--cluster-order",
        choices=["p_desc", "p_full_desc", "size_desc", "root"],
        default="root",
        help="Order clusters on the x-axis and in the saved/printed table.",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=200,
        help="Print at most this many TSV rows to stdout. Use -1 to print all rows.",
    )
    parser.add_argument(
        "--condition-eps",
        type=float,
        nargs="+",
        default=[5.0, 1.0, 0.1],
        help="Thresholds for counting high-condition clusters and hybrid full-attention budget.",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="exact",
        help=(
            "How to compute delta_C. exact uses all q dot K_i scores; "
            "range_bound uses per-dimension K min/max to upper-bound delta_C."
        ),
    )
    return parser.parse_args()


def _build_routing_pos_list(args):
    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty pos_list after applying --pos-start/--pos-end")
    return pos_list


def _resolve_query_positions(args, pos_list):
    pos_set = set(pos_list)
    if args.queries is not None and len(args.queries) > 0:
        queries = [int(x) for x in args.queries]
    else:
        query_start = args.pos_start if args.query_start is None else args.query_start
        query_end_default = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
        query_end = query_end_default if args.query_end is None else args.query_end
        if query_start < 0 or query_end > args.seq_len or query_end <= query_start:
            raise ValueError("--query-start/--query-end must define a non-empty range inside seq_len")
        queries = _choose_evenly(list(range(query_start, query_end)), args.num_queries)

    missing = [pos for pos in queries if pos not in pos_set]
    if missing:
        raise ValueError(f"Query positions not in computed pos_list: {missing[:8]}")
    return queries


def _iter_clusters(row_root, total_available):
    roots = torch.unique(row_root[:total_available]).tolist()
    clusters = []
    for root in roots:
        root = int(root)
        members = (row_root[:total_available] == root).nonzero(as_tuple=False).squeeze(-1)
        if members.numel() > 0:
            clusters.append((root, members))
    return clusters


def _order_rows(rows, order):
    if order == "p_desc":
        return sorted(rows, key=lambda row: (-row["p"], row["root"]))
    if order == "p_full_desc":
        return sorted(rows, key=lambda row: (-row["p_full"], row["root"]))
    if order == "size_desc":
        return sorted(rows, key=lambda row: (-row["size"], row["root"]))
    return sorted(rows, key=lambda row: row["root"])


def _compute_range_bound_delta(q_float, k_cluster, s_c, scale):
    k_max = k_cluster.max(dim=0).values
    k_min = k_cluster.min(dim=0).values
    upper_terms = torch.maximum(q_float * k_max, q_float * k_min)
    lower_terms = torch.minimum(q_float * k_max, q_float * k_min)
    upper_score = upper_terms.sum() / scale
    lower_score = lower_terms.sum() / scale
    return torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())


def _select_delta(exact_delta, range_bound_delta, mode):
    if mode == "exact":
        return exact_delta

    if mode == "range_bound":
        return range_bound_delta

    raise ValueError(f"Unknown delta mode: {mode}")


def _cluster_condition_rows_and_sanity(
    q, k_head, v_head, row_root, total_available, head, query_pos, order, delta_mode
):
    clusters = _iter_clusters(row_root, total_available)
    if len(clusters) == 0:
        return [], None

    scale = math.sqrt(q.numel())
    raw_rows = []
    s_vals = []
    full_logz_vals = []
    sizes = []
    v_bars = []

    for root, members in clusters:
        k_cluster = k_head[members].float()
        v_cluster = v_head[members].float()
        k_bar = k_cluster.mean(dim=0)
        v_bar = v_cluster.mean(dim=0)
        q_float = q.float()
        qk_cluster = torch.mv(k_cluster, q_float) / scale
        s_c = torch.dot(q_float, k_bar.float()) / scale
        centered_scores = qk_cluster - s_c
        exact_delta = centered_scores.abs().max()
        range_bound_delta = _compute_range_bound_delta(
            q_float=q_float,
            k_cluster=k_cluster,
            s_c=s_c,
            scale=scale,
        )
        delta = _select_delta(exact_delta, range_bound_delta, delta_mode)
        b_c = torch.norm(v_cluster, p=2, dim=-1).max()
        exact_delta_value = float(exact_delta.item())
        range_bound_delta_value = float(range_bound_delta.item())
        raw_rows.append(
            {
                "head": int(head),
                "query_pos": int(query_pos),
                "root": int(root),
                "size": int(members.numel()),
                "s": float(s_c.item()),
                "delta": float(delta.item()),
                "delta_exact": exact_delta_value,
                "delta_range_bound": range_bound_delta_value,
                "delta_bound_ratio": range_bound_delta_value / max(exact_delta_value, 1e-30),
                "delta_mode": delta_mode,
                "B": float(b_c.item()),
                "member_min": int(members.min().item()),
                "member_max": int(members.max().item()),
            }
        )
        s_vals.append(s_c)
        full_logz_vals.append(torch.logsumexp(qk_cluster, dim=0))
        sizes.append(float(members.numel()))
        v_bars.append(v_bar)

    s_tensor = torch.stack(s_vals).float()
    size_tensor = torch.tensor(sizes, device=s_tensor.device, dtype=torch.float32)
    z_logits = torch.log(size_tensor) + s_tensor
    p_tensor = torch.softmax(z_logits, dim=0)
    p_full_tensor = torch.softmax(torch.stack(full_logz_vals).float(), dim=0)
    delta_tensor = torch.tensor(
        [row["delta"] for row in raw_rows], device=s_tensor.device, dtype=torch.float32
    )
    exact_delta_tensor = torch.tensor(
        [row["delta_exact"] for row in raw_rows], device=s_tensor.device, dtype=torch.float32
    )
    range_bound_delta_tensor = torch.tensor(
        [row["delta_range_bound"] for row in raw_rows],
        device=s_tensor.device,
        dtype=torch.float32,
    )
    b_c_tensor = torch.tensor(
        [row["B"] for row in raw_rows], device=s_tensor.device, dtype=torch.float32
    )
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    v_bar_tensor = torch.stack(v_bars).float()
    approx_output = (p_tensor.unsqueeze(-1) * v_bar_tensor).sum(dim=0)

    visible_k = k_head[:total_available].float()
    visible_v = v_head[:total_available].float()
    full_logits = torch.mv(visible_k, q.float()) / scale
    full_alpha = torch.softmax(full_logits, dim=0)
    full_output = (full_alpha.unsqueeze(-1) * visible_v).sum(dim=0)

    output_l2 = torch.norm(full_output - approx_output, p=2)
    b_all = torch.norm(visible_v, p=2, dim=-1).max()
    condition_tensor = p_tensor * (
        2.0 * b_all * (torch.cosh(delta_tensor) - 1.0) / denom
        + 2.0 * b_c_tensor * torch.tanh(delta_tensor / 2.0)
    )
    exact_denom = (p_tensor * torch.cosh(exact_delta_tensor)).sum().clamp_min(1e-30)
    exact_condition_tensor = p_tensor * (
        2.0 * b_all * (torch.cosh(exact_delta_tensor) - 1.0) / exact_denom
        + 2.0 * b_c_tensor * torch.tanh(exact_delta_tensor / 2.0)
    )
    range_bound_denom = (
        p_tensor * torch.cosh(range_bound_delta_tensor)
    ).sum().clamp_min(1e-30)
    range_bound_condition_tensor = p_tensor * (
        2.0 * b_all * (torch.cosh(range_bound_delta_tensor) - 1.0) / range_bound_denom
        + 2.0 * b_c_tensor * torch.tanh(range_bound_delta_tensor / 2.0)
    )
    condition_sum = condition_tensor.sum()
    exact_condition_sum = exact_condition_tensor.sum()
    range_bound_condition_sum = range_bound_condition_tensor.sum()
    output_l2_over_2b = output_l2 / (2.0 * b_all).clamp_min(1e-30)

    rows = []
    for cluster_rank, row in enumerate(raw_rows):
        row = dict(row)
        row["Z"] = float(torch.exp(z_logits[cluster_rank]).item())
        row["p"] = float(p_tensor[cluster_rank].item())
        row["p_full"] = float(p_full_tensor[cluster_rank].item())
        row["condition"] = float(condition_tensor[cluster_rank].item())
        row["condition_exact"] = float(exact_condition_tensor[cluster_rank].item())
        row["condition_range_bound"] = float(
            range_bound_condition_tensor[cluster_rank].item()
        )
        rows.append(row)

    ordered = _order_rows(rows, order)
    for rank, row in enumerate(ordered):
        row["cluster_rank"] = rank
    sanity = {
        "head": int(head),
        "query_pos": int(query_pos),
        "clusters": int(len(clusters)),
        "output_l2": float(output_l2.item()),
        "condition_sum": float(condition_sum.item()),
        "condition_exact_sum": float(exact_condition_sum.item()),
        "condition_range_bound_sum": float(range_bound_condition_sum.item()),
        "delta_bound_violations": int(
            (range_bound_delta_tensor + 1e-6 < exact_delta_tensor).sum().item()
        ),
        "delta_mode": delta_mode,
        "B": float(b_all.item()),
        "output_l2_over_2B": float(output_l2_over_2b.item()),
        "output_l2_le_condition_sum": bool(output_l2.item() <= condition_sum.item()),
        "output_l2_over_condition_sum": float(
            output_l2.item() / max(condition_sum.item(), 1e-30)
        ),
    }
    return ordered, sanity


def _save_condition_tsv(out_path, rows):
    columns = [
        "head",
        "query_pos",
        "cluster_rank",
        "root",
        "size",
        "member_min",
        "member_max",
        "s",
        "Z",
        "p",
        "p_full",
        "delta",
        "delta_exact",
        "delta_range_bound",
        "delta_bound_ratio",
        "delta_mode",
        "B",
        "condition",
        "condition_exact",
        "condition_range_bound",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.3g}")
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")
    return columns


def _save_sanity_tsv(out_path, rows):
    columns = [
        "head",
        "query_pos",
        "clusters",
        "output_l2",
        "condition_sum",
        "condition_exact_sum",
        "condition_range_bound_sum",
        "delta_bound_violations",
        "delta_mode",
        "B",
        "output_l2_over_2B",
        "output_l2_le_condition_sum",
        "output_l2_over_condition_sum",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.3g}")
                elif isinstance(val, bool):
                    vals.append(str(int(val)))
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")
    return columns


def _print_sanity_summary(rows):
    if len(rows) == 0:
        print("===== Condition sanity =====")
        print("No sanity rows.")
        return
    output_l2 = torch.tensor([row["output_l2"] for row in rows], dtype=torch.float32)
    bound = torch.tensor([row["condition_sum"] for row in rows], dtype=torch.float32)
    exact_bound = torch.tensor(
        [row["condition_exact_sum"] for row in rows], dtype=torch.float32
    )
    range_bound = torch.tensor(
        [row["condition_range_bound_sum"] for row in rows], dtype=torch.float32
    )
    passed = sum(int(row["output_l2_le_condition_sum"]) for row in rows)
    delta_violations = sum(int(row["delta_bound_violations"]) for row in rows)
    condition_sum_violations = sum(
        int(row["condition_range_bound_sum"] + 1e-6 < row["condition_exact_sum"])
        for row in rows
    )
    print("===== Condition sanity =====")
    print(
        f"output_l2 <= condition_sum: {passed}/{len(rows)} "
        f"({passed / len(rows):.1%})"
    )
    print(
        f"mean output_l2={float(output_l2.mean().item()):.3g}, "
        f"mean condition_sum={float(bound.mean().item()):.3g}, "
        f"max output_l2={float(output_l2.max().item()):.3g}, "
        f"max condition_sum={float(bound.max().item()):.3g}"
    )
    print(
        f"delta range-bound violations: {delta_violations}; "
        f"condition'_sum < condition_sum violations: {condition_sum_violations}"
    )
    print(
        f"mean exact condition_sum={float(exact_bound.mean().item()):.3g}, "
        f"mean range-bound condition'_sum={float(range_bound.mean().item()):.3g}"
    )


def _condition_eps_values(args):
    eps_values = getattr(args, "condition_eps", [5.0, 1.0, 0.1])
    return sorted({float(eps) for eps in eps_values}, reverse=True)


def _collect_condition_eps_budget_rows(rows_by_head_query, head_labels, query_positions, args):
    eps_values = _condition_eps_values(args)
    budget_rows = []
    for head in head_labels:
        head_rows = rows_by_head_query[int(head)]
        for query_pos in query_positions:
            rows = head_rows.get(int(query_pos), [])
            if len(rows) == 0:
                continue

            total_available = int(query_pos) + 1
            total_clusters = len(rows)
            for eps in eps_values:
                selected = [row for row in rows if row["condition"] > eps]
                selected_tokens = sum(int(row["size"]) for row in selected)
                selected_clusters = len(selected)
                hybrid_tokens = selected_tokens + (total_clusters - selected_clusters)
                budget_rows.append(
                    {
                        "head": int(head),
                        "query_pos": int(query_pos),
                        "eps": float(eps),
                        "clusters_gt_eps": int(selected_clusters),
                        "tokens_in_clusters_gt_eps": int(selected_tokens),
                        "hybrid_tokens": int(hybrid_tokens),
                        "hybrid_budget_seq": float(hybrid_tokens / args.seq_len),
                        "hybrid_budget_visible": float(hybrid_tokens / total_available),
                    }
                )
    return budget_rows


def _save_condition_eps_budget_tsv(out_path, rows):
    columns = [
        "head",
        "query_pos",
        "eps",
        "clusters_gt_eps",
        "tokens_in_clusters_gt_eps",
        "hybrid_tokens",
        "hybrid_budget_seq",
        "hybrid_budget_visible",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")
    return columns


def _print_condition_eps_budget_summary(rows):
    if len(rows) == 0:
        print("===== Condition eps budget =====")
        print("No eps budget rows.")
        return

    print("===== Condition eps budget =====")
    eps_values = sorted({row["eps"] for row in rows}, reverse=True)
    for eps in eps_values:
        eps_rows = [row for row in rows if row["eps"] == eps]
        clusters = torch.tensor(
            [row["clusters_gt_eps"] for row in eps_rows], dtype=torch.float32
        )
        selected_tokens = torch.tensor(
            [row["tokens_in_clusters_gt_eps"] for row in eps_rows], dtype=torch.float32
        )
        hybrid_tokens = torch.tensor(
            [row["hybrid_tokens"] for row in eps_rows], dtype=torch.float32
        )
        budget_seq = torch.tensor(
            [row["hybrid_budget_seq"] for row in eps_rows], dtype=torch.float32
        )
        budget_visible = torch.tensor(
            [row["hybrid_budget_visible"] for row in eps_rows], dtype=torch.float32
        )
        print(
            f"eps={eps:g}: "
            f"mean clusters>{eps:g}={float(clusters.mean().item()):.3g}, "
            f"mean selected_tokens={float(selected_tokens.mean().item()):.3g}, "
            f"mean hybrid_tokens={float(hybrid_tokens.mean().item()):.3g}, "
            f"mean hybrid_budget_seq={float(budget_seq.mean().item()):.3g}, "
            f"mean hybrid_budget_visible={float(budget_visible.mean().item()):.3g}"
        )


def _plot_head_conditions(out_path, query_rows, head, query_positions, title, dpi):
    metric_specs = [
        ("p", "P_C", "#2563eb"),
        ("p_full", "P_full_C", "#0891b2"),
        ("condition", "condition", "#059669"),
        ("delta", "delta_C", "#dc2626"),
        ("B", "B_C", "#ea580c"),
        ("size", "cluster size", "#7c3aed"),
    ]
    nrows = max(1, len(query_positions))
    ncols = len(metric_specs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.25 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, query_pos in enumerate(query_positions):
        rows = query_rows.get(int(query_pos), [])
        if len(rows) == 0:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        x = list(range(len(rows)))
        for col_idx, (key, label, color) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if key == "delta":
                vals = [row["delta_exact"] for row in rows]
                approx_vals = [row["delta_range_bound"] for row in rows]
                ax.plot(x, vals, color=color, linewidth=1.1, label="exact")
                ax.plot(
                    x,
                    approx_vals,
                    color="#991b1b",
                    linewidth=1.0,
                    linestyle="--",
                    label="approx",
                )
                ax.legend(fontsize=7, frameon=False)
            elif key == "condition":
                vals = [row["condition_exact"] for row in rows]
                approx_vals = [row["condition_range_bound"] for row in rows]
                ax.plot(x, vals, color=color, linewidth=1.1, label="condition")
                ax.plot(
                    x,
                    approx_vals,
                    color="#166534",
                    linewidth=1.0,
                    linestyle="--",
                    label="condition'",
                )
                ax.legend(fontsize=7, frameon=False)
            else:
                vals = [row[key] for row in rows]
                ax.plot(x, vals, color=color, linewidth=1.1)
            ax.set_title(f"h{head} q={query_pos} {label}")
            ax.set_xlabel("cluster")
            ax.set_ylabel(label)
            ax.grid(alpha=0.22)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_delta_fit(out_path, query_rows, head, query_positions, title, dpi):
    exact_vals = []
    range_vals = []
    colors = []
    cmap = plt.get_cmap("tab10")
    for query_idx, query_pos in enumerate(query_positions):
        rows = query_rows.get(int(query_pos), [])
        exact_vals.extend(row["delta_exact"] for row in rows)
        range_vals.extend(row["delta_range_bound"] for row in rows)
        colors.extend([cmap(query_idx % 10)] * len(rows))

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    if len(exact_vals) == 0:
        ax.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.scatter(exact_vals, range_vals, s=12, alpha=0.58, c=colors, linewidths=0)
        max_val = max(max(exact_vals), max(range_vals))
        min_val = min(min(exact_vals), min(range_vals), 0.0)
        ax.plot([min_val, max_val], [min_val, max_val], color="#111827", linewidth=1.0)
        ax.set_xlabel("exact delta")
        ax.set_ylabel("range-bound delta")
        ax.set_title(f"h{head} delta fit")
        ax.grid(alpha=0.22)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_sanity_curve(out_path, sanity_rows, head_labels, query_positions, title, dpi):
    rows_by_head = {int(head): {} for head in head_labels}
    for row in sanity_rows:
        rows_by_head[int(row["head"])][int(row["query_pos"])] = row

    nrows = max(1, len(head_labels))
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(8.2, 2.7 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, head in enumerate(head_labels):
        ax = axes[row_idx, 0]
        head_rows = rows_by_head.get(int(head), {})
        x = [int(pos) for pos in query_positions if int(pos) in head_rows]
        if len(x) == 0:
            ax.text(0.5, 0.5, "No sanity rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        output_l2 = [head_rows[pos]["output_l2"] for pos in x]
        bound = [head_rows[pos]["condition_sum"] for pos in x]
        ax.plot(x, output_l2, marker="o", linewidth=1.2, label="|o1-o2|", color="#dc2626")
        ax.plot(
            x,
            bound,
            marker="o",
            linewidth=1.2,
            label="condition",
            color="#2563eb",
        )
        ax.set_title(f"h{int(head)} sanity bound")
        ax.set_xlabel("query")
        ax.set_ylabel("L2")
        ax.grid(alpha=0.22)
        ax.legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def collect_condition_stats(
    *,
    out_dir,
    layer_ctx,
    route_mask,
    belong_root,
    pos_list,
    head_labels,
    query_positions,
    args,
):
    del route_mask
    os.makedirs(out_dir, exist_ok=True)
    pos_to_i = {pos: i for i, pos in enumerate(pos_list)}

    ctx_device = layer_ctx.device
    q_all = layer_ctx.rope_qkv[args.layer]["q"].to(ctx_device)[0][head_labels].float()
    k_all = layer_ctx.rope_qkv[args.layer]["k"].to(ctx_device)[0][head_labels].float()
    v_all = layer_ctx.rope_qkv[args.layer]["v"].to(ctx_device)[0][head_labels].float()

    all_rows = []
    sanity_rows = []
    rows_by_head_query = {int(head): {} for head in head_labels}
    tensor_rows = []

    for head_ord, head in enumerate(head_labels):
        for query_pos in query_positions:
            i = pos_to_i[int(query_pos)]
            rows, sanity = _cluster_condition_rows_and_sanity(
                q=q_all[head_ord, int(query_pos)],
                k_head=k_all[head_ord],
                v_head=v_all[head_ord],
                row_root=belong_root[head_ord, i],
                total_available=int(query_pos) + 1,
                head=head,
                query_pos=query_pos,
                order=args.cluster_order,
                delta_mode=getattr(args, "delta_mode", "exact"),
            )
            rows_by_head_query[int(head)][int(query_pos)] = rows
            all_rows.extend(rows)
            if sanity is not None:
                sanity_rows.append(sanity)
            tensor_rows.append(
                {
                    "head": int(head),
                    "query_pos": int(query_pos),
                    "rows": rows,
                    "sanity": sanity,
                }
            )

    table_path = os.path.join(out_dir, "condition_table.tsv")

    sanity_path = os.path.join(out_dir, "condition_sanity.tsv")
    _save_sanity_tsv(sanity_path, sanity_rows)
    _print_sanity_summary(sanity_rows)

    eps_budget_rows = _collect_condition_eps_budget_rows(
        rows_by_head_query=rows_by_head_query,
        head_labels=head_labels,
        query_positions=query_positions,
        args=args,
    )
    eps_budget_path = os.path.join(out_dir, "condition_eps_budget.tsv")
    _save_condition_eps_budget_tsv(eps_budget_path, eps_budget_rows)
    _print_condition_eps_budget_summary(eps_budget_rows)

    sanity_curve_path = os.path.join(out_dir, "condition_sanity_curve.png")
    _plot_sanity_curve(
        out_path=sanity_curve_path,
        sanity_rows=sanity_rows,
        head_labels=head_labels,
        query_positions=query_positions,
        title=f"Layer {args.layer} condition sanity; budget={args.budget:g}",
        dpi=args.plot_dpi,
    )

    plot_paths = {}
    delta_fit_paths = {}
    for head in head_labels:
        plot_path = os.path.join(out_dir, f"condition_head{int(head)}.png")
        _plot_head_conditions(
            out_path=plot_path,
            query_rows=rows_by_head_query[int(head)],
            head=int(head),
            query_positions=query_positions,
            title=(
                f"Layer {args.layer} head {int(head)} condition stats; "
                f"budget={args.budget:g}, order={args.cluster_order}"
            ),
            dpi=args.plot_dpi,
        )
        plot_paths[int(head)] = plot_path
        delta_fit_path = os.path.join(out_dir, f"delta_fit_head{int(head)}.png")
        _plot_delta_fit(
            out_path=delta_fit_path,
            query_rows=rows_by_head_query[int(head)],
            head=int(head),
            query_positions=query_positions,
            title=(
                f"Layer {args.layer} head {int(head)} delta fit; "
                f"budget={args.budget:g}, order={args.cluster_order}"
            ),
            dpi=args.plot_dpi,
        )
        delta_fit_paths[int(head)] = delta_fit_path

    tensor_path = os.path.join(out_dir, "condition_stats.pt")
    torch.save(
        {
            "config": vars(args),
            "layer": int(args.layer),
            "heads": [int(x) for x in head_labels],
            "query_positions": [int(x) for x in query_positions],
            "rows": tensor_rows,
            "sanity_rows": sanity_rows,
            "eps_budget_rows": eps_budget_rows,
            "sanity_curve_path": sanity_curve_path,
        },
        tensor_path,
    )

    print("===== Condition summary =====")
    print(
        f"layer={args.layer}, heads={head_labels}, queries={query_positions}, "
        f"rows={len(all_rows)}"
    )
    print(f"Saved condition table to: {table_path}")
    print(f"Saved condition sanity to: {sanity_path}")
    print(f"Saved condition eps budget to: {eps_budget_path}")
    print(f"Saved condition sanity curve to: {sanity_curve_path}")
    print(f"Saved condition plots to: {list(plot_paths.values())}")
    print(f"Saved delta fit plots to: {list(delta_fit_paths.values())}")
    print(f"Saved condition tensors to: {tensor_path}")

    return {
        "table_path": table_path,
        "sanity_path": sanity_path,
        "eps_budget_path": eps_budget_path,
        "sanity_curve_path": sanity_curve_path,
        "plot_paths": plot_paths,
        "delta_fit_paths": delta_fit_paths,
        "tensor_path": tensor_path,
    }


def _prepare_prefix_patches(args, ctx, routing_pos_list, model_inputs):
    if args.prefix_mode == "full_attention":
        return {}
    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
        return _align_prefix_patches_to_pos_list(prefix_patches, routing_pos_list, args.seq_len)

    return build_baseline_prefix_patches(
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


def main():
    set_seed(42)
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.print_rows < -1:
        raise ValueError("--print-rows must be >= -1")

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
    query_positions = _resolve_query_positions(args, routing_pos_list)
    print(
        f"Build routing clusters with pos_list=[{routing_pos_list[0]}, {routing_pos_list[-1]}] "
        f"({len(routing_pos_list)} positions); condition queries={query_positions}"
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(
        args=args,
        head_idx=head_idx,
        compare_tag="condition",
        include_loss_type=True,
    )

    prefix_patches = _prepare_prefix_patches(args, ctx, routing_pos_list, model_inputs)
    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=routing_pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong, count = gen_mask_h2o_with_belong_all(
        ctx=layer_ctx,
        layer_idx=args.layer,
        pos_list=routing_pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        merge_metric=args.merge_metric,
    )
    del count
    belong_root = canonicalize_belong(belong, routing_pos_list)

    result = collect_condition_stats(
        out_dir=output_dir,
        layer_ctx=layer_ctx,
        route_mask=route_mask,
        belong_root=belong_root,
        pos_list=routing_pos_list,
        head_labels=head_idx,
        query_positions=query_positions,
        args=args,
    )

    summary_path = os.path.join(output_dir, "condition_summary.pt")
    torch.save(
        {
            "config": vars(args),
            "layer": int(args.layer),
            "heads": head_idx,
            "pos_list": routing_pos_list,
            "query_positions": query_positions,
            "outputs": result,
            "belong": belong.detach().cpu(),
            "belong_root": belong_root.detach().cpu(),
        },
        summary_path,
    )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
