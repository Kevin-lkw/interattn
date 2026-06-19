"""
Correlate contiguous-block cluster approximation error with the condition term.

For each selected query head, query position, and block cluster C, this script
computes

    o_C      = sum_{i in C} softmax(qK)_i V_i
    hat{o_C}= p_C mean(V_C), where p_C uses the mean-K block approximation

and compares ||o_C - hat{o_C}||_2 against the cluster condition from
runner_cond_block.py / condition_block.py.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch

from .condition import _choose_evenly, _resolve_query_positions, _build_routing_pos_list
from .condition_block import _resolve_block_size
from .compare_utils import add_common_compare_args, resolve_head_indices, validate_common_args
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .runner_utils import set_seed, str_to_torch_dtype
from .sanity import grouped_query_heads, move_model_inputs_to_device


def _model_output_name(model):
    return str(model).rstrip("/").split("/")[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cluster approximation error vs condition for block clusters."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["block"],
        default_strategy="block",
        include_loss_type=False,
        include_plot_dpi=True,
        prefix_mode_default="full_attention",
        prefix_mode_choices=("full_attention",),
        prefix_mode_help="Only full_attention is used; no previous-layer patches are applied.",
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
        help="Number of query positions to sample when --queries is not set.",
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
        help="When --head/--heads is not set, choose this many heads evenly.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override default block size round(1 / budget).",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
        help="Condition delta mode to use for the main condition column.",
    )
    parser.add_argument(
        "--min-condition",
        type=float,
        default=0.0,
        help="Drop rows with condition <= this value from correlation/plots.",
    )
    return parser.parse_args()


def _resolve_output_dir(args, head_idx):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        if len(head_idx) == 1:
            head_tag = f"head{head_idx[0]}"
        else:
            head_tag = f"heads_{len(head_idx)}"
        llama_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out_dir = os.path.join(
            llama_dir,
            "result",
            _model_output_name(args.model),
            f"{args.dataset}_{args.start}",
            "condition_block_corr",
            f"layer{args.layer}_{head_tag}",
            f"budget={args.budget:g}",
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _range_bound_delta(q_float, k_cluster, s_c, scale):
    k_max = k_cluster.max(dim=0).values
    k_min = k_cluster.min(dim=0).values
    upper = torch.maximum(q_float * k_max, q_float * k_min).sum() / scale
    lower = torch.minimum(q_float * k_max, q_float * k_min).sum() / scale
    return torch.maximum((upper - s_c).abs(), (lower - s_c).abs())


def _condition_from_delta(p_tensor, delta_tensor, b_c_tensor, b_all):
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    return p_tensor * (
        2.0 * b_all * (torch.cosh(delta_tensor) - 1.0) / denom
        + 2.0 * b_c_tensor * torch.tanh(delta_tensor / 2.0)
    )


def _cluster_rows_for_head_query(q, k_head, v_head, query_head, kv_head, query_pos, block_size, delta_mode):
    total_available = int(query_pos) + 1
    scale = math.sqrt(q.numel())
    q_float = q.float()
    visible_k = k_head[:total_available].float()
    visible_v = v_head[:total_available].float()
    full_logits = torch.mv(visible_k, q_float) / scale
    full_alpha = torch.softmax(full_logits, dim=0)
    b_all = torch.norm(visible_v, p=2, dim=-1).max()

    raw = []
    z_logits = []
    full_logz = []
    exact_deltas = []
    range_deltas = []
    b_cs = []

    for start in range(0, total_available, block_size):
        end = min(start + block_size, total_available)
        members = torch.arange(start, end, device=k_head.device)
        k_cluster = k_head[members].float()
        v_cluster = v_head[members].float()
        k_bar = k_cluster.mean(dim=0)
        v_bar = v_cluster.mean(dim=0)
        qk_cluster = torch.mv(k_cluster, q_float) / scale
        s_c = torch.dot(q_float, k_bar) / scale
        exact_delta = (qk_cluster - s_c).abs().max()
        range_delta = _range_bound_delta(q_float, k_cluster, s_c, scale)
        cluster_full_alpha = full_alpha[members]
        o_c = (cluster_full_alpha.unsqueeze(-1) * v_cluster).sum(dim=0)

        raw.append(
            {
                "head": int(query_head),
                "kv_head": int(kv_head),
                "query_pos": int(query_pos),
                "cluster_rank": int(len(raw)),
                "root": int(start),
                "member_min": int(start),
                "member_max": int(end - 1),
                "size": int(end - start),
                "s": float(s_c.item()),
                "full_mass": float(cluster_full_alpha.sum().item()),
                "v_bar": v_bar,
                "o_c": o_c,
            }
        )
        z_logits.append(math.log(end - start) + s_c)
        full_logz.append(torch.logsumexp(qk_cluster, dim=0))
        exact_deltas.append(exact_delta)
        range_deltas.append(range_delta)
        b_cs.append(torch.norm(v_cluster, p=2, dim=-1).max())

    z_tensor = torch.stack(z_logits).float()
    p_tensor = torch.softmax(z_tensor, dim=0)
    p_full_tensor = torch.softmax(torch.stack(full_logz).float(), dim=0)
    exact_delta_tensor = torch.stack(exact_deltas).float()
    range_delta_tensor = torch.stack(range_deltas).float()
    b_c_tensor = torch.stack(b_cs).float()
    condition_exact = _condition_from_delta(p_tensor, exact_delta_tensor, b_c_tensor, b_all)
    condition_range = _condition_from_delta(p_tensor, range_delta_tensor, b_c_tensor, b_all)
    condition = condition_exact if delta_mode == "exact" else condition_range

    rows = []
    for idx, row in enumerate(raw):
        hat_o_c = p_tensor[idx] * row.pop("v_bar")
        o_c = row.pop("o_c")
        error_l2 = torch.norm(o_c - hat_o_c, p=2)
        out = dict(row)
        out.update(
            {
                "p": float(p_tensor[idx].item()),
                "p_full": float(p_full_tensor[idx].item()),
                "delta_exact": float(exact_delta_tensor[idx].item()),
                "delta_range_bound": float(range_delta_tensor[idx].item()),
                "condition": float(condition[idx].item()),
                "condition_exact": float(condition_exact[idx].item()),
                "condition_range_bound": float(condition_range[idx].item()),
                "cluster_error_l2": float(error_l2.item()),
                "error_over_condition": float(
                    error_l2.item() / max(float(condition[idx].item()), 1e-30)
                ),
            }
        )
        rows.append(out)
    return rows


def _rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _pearson(x, y):
    if x.numel() < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.norm(xc) * torch.norm(yc)
    if float(denom.item()) <= 0:
        return float("nan")
    return float((xc * yc).sum().div(denom).item())


def _summarize(rows):
    kept = [
        row for row in rows
        if row["condition"] > 0 and row["cluster_error_l2"] >= 0
    ]
    if not kept:
        return {"n": 0}
    x = torch.tensor([row["condition"] for row in kept], dtype=torch.float64)
    y = torch.tensor([row["cluster_error_l2"] for row in kept], dtype=torch.float64)
    slope_origin = float((x * y).sum().div((x * x).sum().clamp_min(1e-300)).item())
    y_hat = slope_origin * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-300)
    log_x = torch.log10(x.clamp_min(1e-300))
    log_y = torch.log10(y.clamp_min(1e-300))
    log_xc = log_x - log_x.mean()
    log_slope = float((log_xc * (log_y - log_y.mean())).sum().div((log_xc ** 2).sum().clamp_min(1e-300)).item())
    log_intercept = float((log_y.mean() - log_slope * log_x.mean()).item())
    return {
        "n": int(len(kept)),
        "pearson": _pearson(x.float(), y.float()),
        "spearman": _pearson(_rankdata(x.float()), _rankdata(y.float())),
        "log10_pearson": _pearson(log_x.float(), log_y.float()),
        "slope_origin_error_per_condition": slope_origin,
        "r2_origin": float((1.0 - ss_res / ss_tot).item()),
        "log10_slope": log_slope,
        "log10_intercept": log_intercept,
        "mean_error_over_condition": float(
            torch.tensor([row["error_over_condition"] for row in kept]).mean().item()
        ),
    }


def _write_rows(path, rows):
    cols = [
        "head",
        "kv_head",
        "query_pos",
        "cluster_rank",
        "root",
        "member_min",
        "member_max",
        "size",
        "s",
        "p",
        "p_full",
        "full_mass",
        "delta_exact",
        "delta_range_bound",
        "condition",
        "condition_exact",
        "condition_range_bound",
        "cluster_error_l2",
        "error_over_condition",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.8e}" if isinstance(val, float) else str(val))
            f.write("\t".join(vals) + "\n")


def _write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}\t{value}\n")


def _plot_scatter(path, rows, summary, title, dpi, log_axes=False):
    plot_rows = [row for row in rows if row["condition"] > 0 and row["cluster_error_l2"] > 0]
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    if plot_rows:
        x = torch.tensor([row["condition"] for row in plot_rows], dtype=torch.float32)
        y = torch.tensor([row["cluster_error_l2"] for row in plot_rows], dtype=torch.float32)
        sizes = [max(10, min(70, row["size"] * 4)) for row in plot_rows]
        ax.scatter(x.numpy(), y.numpy(), s=sizes, alpha=0.55, color="#2563eb", edgecolors="none")
        if summary.get("n", 0) > 0:
            x_line = torch.linspace(float(x.min().item()), float(x.max().item()), 100)
            y_line = float(summary["slope_origin_error_per_condition"]) * x_line
            ax.plot(x_line.numpy(), y_line.numpy(), color="#dc2626", linewidth=1.2, label="fit through origin")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No positive rows", ha="center", va="center", transform=ax.transAxes)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("condition")
    ax.set_ylabel("||o_C - hat{o_C}||_2")
    ax.set_title(title)
    ax.grid(alpha=0.24)
    text = (
        f"n={summary.get('n', 0)}\n"
        f"Pearson={summary.get('pearson', float('nan')):.3g}\n"
        f"Spearman={summary.get('spearman', float('nan')):.3g}\n"
        f"log Pearson={summary.get('log10_pearson', float('nan')):.3g}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    set_seed(42)
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    args.block_size = _resolve_block_size(args)

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
    pos_list = _build_routing_pos_list(args)
    query_positions = _resolve_query_positions(args, pos_list)
    output_dir = _resolve_output_dir(args, head_idx)

    print(
        f"Collect layer={args.layer}, heads={head_idx}, queries={query_positions}, "
        f"block_size={args.block_size}, delta_mode={args.delta_mode}"
    )
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch={},
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)
    q_all = layer_ctx.rope_qkv[args.layer]["q"].to(ctx.device)[0].float()
    k_all = layer_ctx.rope_qkv[args.layer]["k"].to(ctx.device)[0].float()
    v_all = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0].float()

    rows = []
    for kv_head, _out_indices, query_heads in grouped_query_heads(
        head_idx,
        ctx.model_config,
        num_kv_heads=k_all.shape[0],
    ):
        for query_head in query_heads:
            for query_pos in query_positions:
                rows.extend(
                    _cluster_rows_for_head_query(
                        q=q_all[query_head, int(query_pos)],
                        k_head=k_all[kv_head],
                        v_head=v_all[kv_head],
                        query_head=query_head,
                        kv_head=kv_head,
                        query_pos=int(query_pos),
                        block_size=args.block_size,
                        delta_mode=args.delta_mode,
                    )
                )

    corr_rows = [row for row in rows if row["condition"] > args.min_condition]
    summary = _summarize(corr_rows)
    summary.update(
        {
            "layer": int(args.layer),
            "heads": ",".join(str(x) for x in head_idx),
            "query_positions": ",".join(str(x) for x in query_positions),
            "block_size": int(args.block_size),
            "delta_mode": args.delta_mode,
            "rows_total": int(len(rows)),
            "rows_after_min_condition": int(len(corr_rows)),
            "min_condition": float(args.min_condition),
        }
    )

    rows_path = os.path.join(output_dir, "cluster_approx_condition.tsv")
    summary_path = os.path.join(output_dir, "cluster_approx_condition_summary.tsv")
    tensor_path = os.path.join(output_dir, "cluster_approx_condition.pt")
    scatter_path = os.path.join(output_dir, "cluster_error_vs_condition.png")
    log_scatter_path = os.path.join(output_dir, "cluster_error_vs_condition_loglog.png")
    _write_rows(rows_path, rows)
    _write_summary(summary_path, summary)
    torch.save({"config": vars(args), "rows": rows, "summary": summary}, tensor_path)
    title = f"Layer {args.layer}: cluster error vs condition"
    _plot_scatter(scatter_path, corr_rows, summary, title, args.plot_dpi, log_axes=False)
    _plot_scatter(log_scatter_path, corr_rows, summary, title + " (log-log)", args.plot_dpi, log_axes=True)

    print("===== Cluster approximation vs condition =====")
    print(
        f"rows={len(rows)} used={summary['rows_after_min_condition']} "
        f"pearson={summary.get('pearson', float('nan')):.4g} "
        f"spearman={summary.get('spearman', float('nan')):.4g} "
        f"log_pearson={summary.get('log10_pearson', float('nan')):.4g} "
        f"slope_origin={summary.get('slope_origin_error_per_condition', float('nan')):.4g}"
    )
    print(f"Saved rows to: {rows_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved plots to: {scatter_path}, {log_scatter_path}")
    print(f"Saved tensors to: {tensor_path}")


if __name__ == "__main__":
    main()
