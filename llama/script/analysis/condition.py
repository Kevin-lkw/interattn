"""
Inspect cluster-level conditioning terms for H2O count-all routing.

For each selected query q and each cluster C at that query:

    s_C     = q dot mean(K_C) / sqrt(d)
    Z_C     = |C| exp(s_C)
    p_C     = Z_C / sum_C' Z_C'
    delta_C = max_i in C |q dot (K_i - mean(K_C)) / sqrt(d)|
    B_C     = max_i in C ||V_i||

The plotted condition term uses p_C as p_hat_C:

    p_C * ((cosh(delta_C) - 1) / sum_C' p_C' cosh(delta_C')
           + tanh(delta_C / 2))
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
        choices=["p_desc", "size_desc", "root"],
        default="p_desc",
        help="Order clusters on the x-axis and in the saved/printed table.",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=200,
        help="Print at most this many TSV rows to stdout. Use -1 to print all rows.",
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
    if order == "size_desc":
        return sorted(rows, key=lambda row: (-row["size"], row["root"]))
    return sorted(rows, key=lambda row: row["root"])


def _cluster_condition_rows(q, k_head, v_head, row_root, total_available, head, query_pos, order):
    clusters = _iter_clusters(row_root, total_available)
    if len(clusters) == 0:
        return []

    scale = math.sqrt(q.numel())
    raw_rows = []
    s_vals = []
    sizes = []

    for root, members in clusters:
        k_cluster = k_head[members].float()
        v_cluster = v_head[members].float()
        k_bar = k_cluster.mean(dim=0)
        centered_scores = torch.mv(k_cluster - k_bar.unsqueeze(0), q.float()) / scale
        delta = centered_scores.abs().max()
        s_c = torch.dot(q.float(), k_bar.float()) / scale
        b_c = torch.norm(v_cluster, p=2, dim=-1).max()
        raw_rows.append(
            {
                "head": int(head),
                "query_pos": int(query_pos),
                "root": int(root),
                "size": int(members.numel()),
                "s": float(s_c.item()),
                "delta": float(delta.item()),
                "B": float(b_c.item()),
                "member_min": int(members.min().item()),
                "member_max": int(members.max().item()),
            }
        )
        s_vals.append(s_c)
        sizes.append(float(members.numel()))

    s_tensor = torch.stack(s_vals).float()
    size_tensor = torch.tensor(sizes, device=s_tensor.device, dtype=torch.float32)
    z_logits = torch.log(size_tensor) + s_tensor
    p_tensor = torch.softmax(z_logits, dim=0)
    delta_tensor = torch.tensor(
        [row["delta"] for row in raw_rows], device=s_tensor.device, dtype=torch.float32
    )
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    condition_tensor = p_tensor * (
        (torch.cosh(delta_tensor) - 1.0) / denom + torch.tanh(delta_tensor / 2.0)
    )

    rows = []
    for cluster_rank, row in enumerate(raw_rows):
        row = dict(row)
        row["Z"] = float(torch.exp(z_logits[cluster_rank]).item())
        row["p"] = float(p_tensor[cluster_rank].item())
        row["condition"] = float(condition_tensor[cluster_rank].item())
        rows.append(row)

    ordered = _order_rows(rows, order)
    for rank, row in enumerate(ordered):
        row["cluster_rank"] = rank
    return ordered


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
        "delta",
        "B",
        "condition",
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


def _print_condition_table(rows, columns, max_rows):
    if max_rows == 0:
        return
    shown = rows if max_rows < 0 else rows[:max_rows]
    print("===== Condition table =====")
    print("\t".join(columns))
    for row in shown:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3g}")
            else:
                vals.append(str(val))
        print("\t".join(vals))
    if max_rows >= 0 and len(rows) > max_rows:
        print(f"... printed {max_rows}/{len(rows)} rows; full table saved to TSV.")


def _plot_head_conditions(out_path, query_rows, head, query_positions, title, dpi):
    metric_specs = [
        ("p", "P_C", "#2563eb"),
        ("condition", "condition", "#059669"),
        ("delta", "delta_C", "#dc2626"),
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
            vals = [row[key] for row in rows]
            ax.plot(x, vals, color=color, linewidth=1.1)
            ax.set_title(f"h{head} q={query_pos} {label}")
            ax.set_xlabel("cluster")
            ax.set_ylabel(label)
            ax.grid(alpha=0.22)

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
    rows_by_head_query = {int(head): {} for head in head_labels}
    tensor_rows = []

    for head_ord, head in enumerate(head_labels):
        for query_pos in query_positions:
            i = pos_to_i[int(query_pos)]
            rows = _cluster_condition_rows(
                q=q_all[head_ord, int(query_pos)],
                k_head=k_all[head_ord],
                v_head=v_all[head_ord],
                row_root=belong_root[head_ord, i],
                total_available=int(query_pos) + 1,
                head=head,
                query_pos=query_pos,
                order=args.cluster_order,
            )
            rows_by_head_query[int(head)][int(query_pos)] = rows
            all_rows.extend(rows)
            tensor_rows.append(
                {
                    "head": int(head),
                    "query_pos": int(query_pos),
                    "rows": rows,
                }
            )

    table_path = os.path.join(out_dir, "condition_table.tsv")
    columns = _save_condition_tsv(table_path, all_rows)
    _print_condition_table(all_rows, columns, args.print_rows)

    plot_paths = {}
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

    tensor_path = os.path.join(out_dir, "condition_stats.pt")
    torch.save(
        {
            "config": vars(args),
            "layer": int(args.layer),
            "heads": [int(x) for x in head_labels],
            "query_positions": [int(x) for x in query_positions],
            "rows": tensor_rows,
        },
        tensor_path,
    )

    print("===== Condition summary =====")
    print(
        f"layer={args.layer}, heads={head_labels}, queries={query_positions}, "
        f"rows={len(all_rows)}"
    )
    print(f"Saved condition table to: {table_path}")
    print(f"Saved condition plots to: {list(plot_paths.values())}")
    print(f"Saved condition tensors to: {tensor_path}")

    return {
        "table_path": table_path,
        "plot_paths": plot_paths,
        "tensor_path": tensor_path,
    }


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

    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
        prefix_patches = _align_prefix_patches_to_pos_list(
            prefix_patches, routing_pos_list, args.seq_len
        )
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
