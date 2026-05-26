"""
Decompose the count-refined H2O-all routing error in unnormalized numerator space.

This script intentionally ignores every softmax denominator.  For each query row it
compares shifted attention numerators under the same row-wise max shift:

    count_all_num       = S_tilde
    key_corrected_num   = S_tilde + key_num_error
    value_corrected_num = S_tilde + value_num_error
    reconstructed_num   = S_tilde + key_num_error + value_num_error
    full_num            = S_full

The reconstructed numerator should match full_num up to numerical precision.
"""

import argparse
import math
import os
import warnings

import matplotlib.pyplot as plt
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


MAX_PERCENTILE_VALUES = 5_000_000
QK_RATIO_MIN_ABS = 1.0

from .attention import gen_mask_h2o_with_belong_all
from .compare_error import canonicalize_belong, get_qk_logits
from .compare_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    plot_per_pos_two_lines,
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
        description="Decompose count-all routing error in unnormalized numerator space."
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
        help="Cluster merge target metric. Default is k; v is allowed only for diagnostics.",
    )
    return parser.parse_args()


def progress_iter(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def decompose_count_all_error_unnorm(qk_logits, v_head, route_mask, belong_root, count, pos_list):
    n_heads, n_pos, seq_len = qk_logits.shape
    d = v_head.shape[-1]
    if route_mask.shape != qk_logits.shape:
        raise ValueError(
            f"route_mask shape mismatch: got {tuple(route_mask.shape)} expected {tuple(qk_logits.shape)}"
        )
    if belong_root.shape != qk_logits.shape:
        raise ValueError(
            f"belong_root shape mismatch: got {tuple(belong_root.shape)} expected {tuple(qk_logits.shape)}"
        )
    if count.shape != qk_logits.shape:
        raise ValueError(f"count shape mismatch: got {tuple(count.shape)} expected {tuple(qk_logits.shape)}")
    if v_head.shape[0] != n_heads or v_head.shape[1] != seq_len:
        raise ValueError(
            f"v_head shape mismatch: got {tuple(v_head.shape)} expected ({n_heads}, {seq_len}, d)"
        )

    count_all_num = torch.zeros(n_heads, n_pos, d, device=qk_logits.device)
    key_num_error = torch.zeros_like(count_all_num)
    value_num_error = torch.zeros_like(count_all_num)
    full_num = torch.zeros_like(count_all_num)

    total_steps = n_heads * n_pos
    outer = ((h, i, pos) for h in range(n_heads) for i, pos in enumerate(pos_list))
    for h, i, pos in progress_iter(
        outer,
        total=total_steps,
        desc="decompose unnorm clusters",
        dynamic_ncols=True,
    ):
        total_available = pos + 1
        row_logits = qk_logits[h, i, :total_available].float()
        row_shift = row_logits.max()
        row_exp = torch.exp(row_logits - row_shift)
        row_v = v_head[h, :total_available].float()
        row_root = belong_root[h, i, :total_available]
        if (row_root < 0).any():
            raise ValueError(f"belong_root contains negative index at head={h}, pos={pos}")

        kept = (~torch.isneginf(route_mask[h, i, :total_available])).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if kept.numel() == 0:
            raise ValueError(f"No kept representatives at head={h}, pos={pos}")

        c = count[h, i, kept].to(torch.float32)
        if (c <= 0).any():
            raise ValueError(f"Kept representative has non-positive count at head={h}, pos={pos}")

        cluster_exp = torch.zeros(total_available, device=qk_logits.device, dtype=torch.float32)
        cluster_exp.scatter_add_(0, row_root, row_exp)

        cluster_count = torch.zeros(total_available, device=qk_logits.device, dtype=torch.long)
        cluster_count.scatter_add_(0, row_root, torch.ones_like(row_root, dtype=torch.long))
        if not torch.equal(cluster_count[kept], count[h, i, kept]):
            raise ValueError(f"count/belong mismatch at head={h}, pos={pos}")

        rep_exp = row_exp[kept]
        v_rep = v_head[h, kept].float()
        weighted_rep_exp = c * rep_exp

        count_all_num[h, i] = (weighted_rep_exp.unsqueeze(-1) * v_rep).sum(0)

        key_exp_gap = cluster_exp[kept] - weighted_rep_exp
        key_num_error[h, i] = (key_exp_gap.unsqueeze(-1) * v_rep).sum(0)

        full_num[h, i] = (row_exp.unsqueeze(-1) * row_v).sum(0)
        value_num_error[h, i] = full_num[h, i] - count_all_num[h, i] - key_num_error[h, i]

    key_corrected_num = count_all_num + key_num_error
    value_corrected_num = count_all_num + value_num_error
    reconstructed_num = count_all_num + key_num_error + value_num_error

    return {
        "count_all_num": count_all_num,
        "key_corrected_num": key_corrected_num,
        "value_corrected_num": value_corrected_num,
        "reconstructed_num": reconstructed_num,
        "full_num": full_num,
        "key_num_error": key_num_error,
        "value_num_error": value_num_error,
    }


def l2_per_pos(x, target):
    return torch.norm(x.float() - target.float(), p=2, dim=-1).mean(dim=0)


def norm_per_pos(x):
    return torch.norm(x.float(), p=2, dim=-1).mean(dim=0)


def cosine_per_pos(x, target, eps=1e-12):
    x_f = x.float()
    target_f = target.float()
    dot = (x_f * target_f).sum(dim=-1)
    denom = torch.norm(x_f, p=2, dim=-1) * torch.norm(target_f, p=2, dim=-1)
    return (dot / denom.clamp_min(eps)).mean(dim=0)


def _finite_values(x):
    x = x.detach().float().cpu()
    return x[torch.isfinite(x)]


def _sample_for_percentile(vals, max_values=MAX_PERCENTILE_VALUES):
    vals = vals.reshape(-1)
    if vals.numel() <= max_values:
        return vals

    step = int(math.ceil(vals.numel() / max_values))
    return vals[::step]


def _summary_stats(x):
    vals = _finite_values(x)
    if vals.numel() == 0:
        return {
            "mean": float("nan"),
            "var": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
        }

    abs_vals = vals.abs()
    abs_pct_vals = _sample_for_percentile(abs_vals)
    return {
        "mean": float(vals.mean().item()),
        "var": float(vals.var(unbiased=False).item()),
        "std": float(vals.std(unbiased=False).item()),
        "median": float(vals.median().item()),
        "p95_abs": _percentile(abs_pct_vals, 0.95),
        "max_abs": float(abs_vals.max().item()),
    }


def _percentile(vals, q):
    vals = vals.reshape(-1)
    if vals.numel() == 0:
        return float("nan")
    if vals.numel() == 1:
        return float(vals[0].item())

    k = int(math.ceil(float(q) * vals.numel()))
    k = min(max(k, 1), vals.numel())
    return float(vals.kthvalue(k).values.item())


def _robust_vlim(x, signed=False, quantile=0.99):
    vals = _finite_values(x)
    if vals.numel() == 0:
        return (-1.0, 1.0) if signed else (0.0, 1.0)

    pct_vals = _sample_for_percentile(vals)
    if signed:
        vmax = _percentile(pct_vals.abs(), quantile)
        vmax = max(vmax, 1e-12)
        return -vmax, vmax

    vmin = _percentile(pct_vals, 1.0 - quantile)
    vmax = _percentile(pct_vals, quantile)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    return vmin, vmax


def _plot_head_heatmaps(out_path, matrix, head_labels, pos_list, title, cmap, signed=False, dpi=180):
    n_heads = matrix.shape[0]
    fig_h = max(3.0, 2.8 * n_heads)
    fig, axes = plt.subplots(n_heads, 1, figsize=(10.5, fig_h), constrained_layout=True)
    if n_heads == 1:
        axes = [axes]

    vmin, vmax = _robust_vlim(matrix, signed=signed)
    x0 = 0
    x1 = matrix.shape[-1] - 1
    y0 = pos_list[-1] if len(pos_list) > 0 else matrix.shape[1] - 1
    y1 = pos_list[0] if len(pos_list) > 0 else 0

    for i, ax in enumerate(axes):
        arr = matrix[i].detach().float().cpu().numpy()
        im = ax.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[x0, x1, y0, y1],
        )
        ax.set_title(f"{title} | head {head_labels[i]}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_matrix_stats_tsv(out_path, head_labels, stats_by_name):
    stat_names = ["mean", "var", "std", "median", "p95_abs", "max_abs"]
    col_width = 14
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{'scope':<10} {'metric':<32}" + "".join(f"{name:>{col_width}}" for name in stat_names) + "\n")
        for metric_name, metric_stats in stats_by_name["all"].items():
            vals = [metric_stats[name] for name in stat_names]
            f.write(
                f"{'all':<10} {metric_name:<32}"
                + "".join(f"{v:>{col_width}.2f}" for v in vals)
                + "\n"
            )
        for i, head in enumerate(head_labels):
            for metric_name, per_head_stats in stats_by_name["head"].items():
                vals = [per_head_stats[i][name] for name in stat_names]
                f.write(
                    f"{f'head{head}':<10} {metric_name:<32}"
                    + "".join(f"{v:>{col_width}.2f}" for v in vals)
                    + "\n"
                )


def visualize_error(
    *,
    out_dir,
    q_head,
    k_head,
    qk_logits,
    belong_root,
    pos_list,
    head_labels,
    dpi=180,
    qk_ratio_min_abs=QK_RATIO_MIN_ABS,
):
    """Visualize delta-K, q.delta-K, and qk magnitudes for routed clusters.

    All dot products use the same /sqrt(head_dim) scale as qk_logits.  Invalid
    non-causal cells are stored as NaN and ignored in printed/TSV statistics.
    """
    n_heads, n_pos, seq_len = qk_logits.shape
    if q_head.shape[:2] != (n_heads, n_pos):
        raise ValueError(f"q_head shape mismatch: got {tuple(q_head.shape)} expected ({n_heads}, {n_pos}, d)")
    if k_head.shape[0] != n_heads or k_head.shape[1] != seq_len:
        raise ValueError(f"k_head shape mismatch: got {tuple(k_head.shape)} expected ({n_heads}, {seq_len}, d)")
    if belong_root.shape != qk_logits.shape:
        raise ValueError(
            f"belong_root shape mismatch: got {tuple(belong_root.shape)} expected {tuple(qk_logits.shape)}"
        )

    scale = float(q_head.shape[-1]) ** 0.5
    nan = float("nan")
    delta_k_l2 = torch.full((n_heads, n_pos, seq_len), nan, device=q_head.device, dtype=torch.float32)
    k_l2 = torch.full_like(delta_k_l2, nan)
    q_delta_k = torch.full_like(delta_k_l2, nan)
    qk_valid = torch.full_like(delta_k_l2, nan)
    rel_delta_k_to_k = torch.full_like(delta_k_l2, nan)
    rel_q_delta_to_qk_raw = torch.full_like(delta_k_l2, nan)
    rel_q_delta_to_qk = torch.full_like(delta_k_l2, nan)

    for h in range(n_heads):
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            roots = belong_root[h, i, :total_available].to(device=k_head.device, dtype=torch.long)
            delta_k = k_head[h, :total_available].float() - k_head[h, roots].float()
            q = q_head[h, i].float()
            q_delta = (delta_k @ q) / scale
            qk = qk_logits[h, i, :total_available].float()

            row_delta_k_l2 = torch.norm(delta_k, p=2, dim=-1)
            row_k_l2 = torch.norm(k_head[h, :total_available].float(), p=2, dim=-1)
            delta_k_l2[h, i, :total_available] = row_delta_k_l2
            k_l2[h, i, :total_available] = row_k_l2
            q_delta_k[h, i, :total_available] = q_delta
            qk_valid[h, i, :total_available] = qk
            rel_delta_k_to_k[h, i, :total_available] = row_delta_k_l2 / row_k_l2.clamp_min(1e-12)
            row_qk_abs = qk.abs()
            row_ratio_raw = q_delta.abs() / row_qk_abs.clamp_min(1e-12)
            rel_q_delta_to_qk_raw[h, i, :total_available] = row_ratio_raw
            rel_q_delta_to_qk[h, i, :total_available] = torch.where(
                row_qk_abs >= qk_ratio_min_abs,
                row_ratio_raw,
                torch.full_like(row_ratio_raw, nan),
            )

    q_delta_abs = q_delta_k.abs()
    qk_abs = qk_valid.abs()
    mean_abs_qk = _summary_stats(qk_abs)["mean"]
    rel_q_delta_to_mean_qk = q_delta_abs / max(mean_abs_qk, 1e-12)
    stats_by_name = {
        "all": {
            "k_l2": _summary_stats(k_l2),
            "delta_k_l2": _summary_stats(delta_k_l2),
            "delta_k_l2_over_k_l2": _summary_stats(rel_delta_k_to_k),
            "q_delta_k": _summary_stats(q_delta_k),
            "abs_q_delta_k": _summary_stats(q_delta_abs),
            "qk": _summary_stats(qk_valid),
            "abs_qk": _summary_stats(qk_abs),
            "abs_qdelta_over_mean_absqk": _summary_stats(rel_q_delta_to_mean_qk),
            "abs_qdelta_over_absqk_qk_ge_1": _summary_stats(rel_q_delta_to_qk),
        },
        "head": {
            "k_l2": [_summary_stats(k_l2[i]) for i in range(n_heads)],
            "delta_k_l2": [_summary_stats(delta_k_l2[i]) for i in range(n_heads)],
            "delta_k_l2_over_k_l2": [_summary_stats(rel_delta_k_to_k[i]) for i in range(n_heads)],
            "q_delta_k": [_summary_stats(q_delta_k[i]) for i in range(n_heads)],
            "abs_q_delta_k": [_summary_stats(q_delta_abs[i]) for i in range(n_heads)],
            "qk": [_summary_stats(qk_valid[i]) for i in range(n_heads)],
            "abs_qk": [_summary_stats(qk_abs[i]) for i in range(n_heads)],
            "abs_qdelta_over_mean_absqk": [
                _summary_stats(rel_q_delta_to_mean_qk[i]) for i in range(n_heads)
            ],
            "abs_qdelta_over_absqk_qk_ge_1": [
                _summary_stats(rel_q_delta_to_qk[i]) for i in range(n_heads)
            ],
        },
    }

    stats_path = os.path.join(out_dir, "delta_k_q_delta_k_stats.tsv")
    _save_matrix_stats_tsv(stats_path, head_labels, stats_by_name)

    tensor_path = os.path.join(out_dir, "delta_k_q_delta_k_matrices.pt")
    torch.save(
        {
            "head_labels": head_labels,
            "pos_list": pos_list,
            "k_l2": k_l2.detach().cpu(),
            "delta_k_l2": delta_k_l2.detach().cpu(),
            "delta_k_l2_over_k_l2": rel_delta_k_to_k.detach().cpu(),
            "q_delta_k": q_delta_k.detach().cpu(),
            "abs_q_delta_k": q_delta_abs.detach().cpu(),
            "qk": qk_valid.detach().cpu(),
            "abs_qk": qk_abs.detach().cpu(),
            "abs_q_delta_k_over_abs_qk_raw": rel_q_delta_to_qk_raw.detach().cpu(),
            "abs_qdelta_over_mean_absqk": rel_q_delta_to_mean_qk.detach().cpu(),
            "abs_qdelta_over_absqk_qk_ge_1": rel_q_delta_to_qk.detach().cpu(),
            "mean_abs_qk": mean_abs_qk,
            "qk_ratio_min_abs": qk_ratio_min_abs,
            "stats": stats_by_name,
        },
        tensor_path,
    )

    k_plot = os.path.join(out_dir, "heatmap_k_l2.png")
    delta_plot = os.path.join(out_dir, "heatmap_delta_k_l2.png")
    rel_delta_plot = os.path.join(out_dir, "heatmap_delta_k_l2_over_k_l2.png")
    q_delta_plot = os.path.join(out_dir, "heatmap_q_delta_k.png")
    q_delta_abs_plot = os.path.join(out_dir, "heatmap_abs_q_delta_k.png")
    qk_plot = os.path.join(out_dir, "heatmap_qk_logits.png")
    qk_abs_plot = os.path.join(out_dir, "heatmap_abs_qk_logits.png")
    rel_plot = os.path.join(out_dir, "heatmap_abs_qdelta_over_absqk_qk_ge_1.png")
    rel_mean_plot = os.path.join(out_dir, "heatmap_abs_qdelta_over_mean_absqk.png")

    _plot_head_heatmaps(
        k_plot,
        k_l2,
        head_labels,
        pos_list,
        title="||K_i||2",
        cmap="viridis",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        delta_plot,
        delta_k_l2,
        head_labels,
        pos_list,
        title="||delta K||2",
        cmap="viridis",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        rel_delta_plot,
        rel_delta_k_to_k,
        head_labels,
        pos_list,
        title="||delta K||2 / ||K_i||2",
        cmap="viridis",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        q_delta_plot,
        q_delta_k,
        head_labels,
        pos_list,
        title="q dot delta K / sqrt(d)",
        cmap="coolwarm",
        signed=True,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        q_delta_abs_plot,
        q_delta_abs,
        head_labels,
        pos_list,
        title="|q dot delta K| / sqrt(d)",
        cmap="magma",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        qk_plot,
        qk_valid,
        head_labels,
        pos_list,
        title="qk logits",
        cmap="coolwarm",
        signed=True,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        qk_abs_plot,
        qk_abs,
        head_labels,
        pos_list,
        title="|qk logits|",
        cmap="magma",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        rel_plot,
        rel_q_delta_to_qk,
        head_labels,
        pos_list,
        title=f"|q dot delta K| / |qk| where |qk| >= {qk_ratio_min_abs:g}",
        cmap="viridis",
        signed=False,
        dpi=dpi,
    )
    _plot_head_heatmaps(
        rel_mean_plot,
        rel_q_delta_to_mean_qk,
        head_labels,
        pos_list,
        title="|q dot delta K| / mean(|qk|)",
        cmap="viridis",
        signed=False,
        dpi=dpi,
    )

    s = stats_by_name["all"]
    print("===== Delta-K / Q-Delta-K Scale Summary =====")
    print(
        "k_l2: "
        f"mean={s['k_l2']['mean']:.2f}, std={s['k_l2']['std']:.2f}, "
        f"var={s['k_l2']['var']:.2f}, p95_abs={s['k_l2']['p95_abs']:.2f}, "
        f"max_abs={s['k_l2']['max_abs']:.2f}"
    )
    print(
        "delta_k_l2: "
        f"mean={s['delta_k_l2']['mean']:.2f}, std={s['delta_k_l2']['std']:.2f}, "
        f"var={s['delta_k_l2']['var']:.2f}, p95_abs={s['delta_k_l2']['p95_abs']:.2f}, "
        f"max_abs={s['delta_k_l2']['max_abs']:.2f}"
    )
    print(
        "delta_k_l2/k_l2: "
        f"mean={s['delta_k_l2_over_k_l2']['mean']:.2f}, "
        f"std={s['delta_k_l2_over_k_l2']['std']:.2f}, "
        f"p95_abs={s['delta_k_l2_over_k_l2']['p95_abs']:.2f}, "
        f"max_abs={s['delta_k_l2_over_k_l2']['max_abs']:.2f}"
    )
    print(
        "q_delta_k: "
        f"mean={s['q_delta_k']['mean']:.2f}, std={s['q_delta_k']['std']:.2f}, "
        f"var={s['q_delta_k']['var']:.2f}, p95_abs={s['q_delta_k']['p95_abs']:.2f}, "
        f"max_abs={s['q_delta_k']['max_abs']:.2f}"
    )
    print(
        "qk logits: "
        f"mean={s['qk']['mean']:.2f}, std={s['qk']['std']:.2f}, "
        f"var={s['qk']['var']:.2f}, p95_abs={s['qk']['p95_abs']:.2f}, "
        f"max_abs={s['qk']['max_abs']:.2f}"
    )
    print(
        "|q_delta_k|/mean(|qk|): "
        f"mean={s['abs_qdelta_over_mean_absqk']['mean']:.2f}, "
        f"std={s['abs_qdelta_over_mean_absqk']['std']:.2f}, "
        f"p95_abs={s['abs_qdelta_over_mean_absqk']['p95_abs']:.2f}, "
        f"max_abs={s['abs_qdelta_over_mean_absqk']['max_abs']:.2f}"
    )
    print(
        f"|q_delta_k|/|qk| where |qk|>={qk_ratio_min_abs:g}: "
        f"mean={s['abs_qdelta_over_absqk_qk_ge_1']['mean']:.2f}, "
        f"std={s['abs_qdelta_over_absqk_qk_ge_1']['std']:.2f}, "
        f"p95_abs={s['abs_qdelta_over_absqk_qk_ge_1']['p95_abs']:.2f}, "
        f"max_abs={s['abs_qdelta_over_absqk_qk_ge_1']['max_abs']:.2f}"
    )

    return {
        "stats_path": stats_path,
        "tensor_path": tensor_path,
        "plots": {
            "k_l2": k_plot,
            "delta_k_l2": delta_plot,
            "delta_k_l2_over_k_l2": rel_delta_plot,
            "q_delta_k": q_delta_plot,
            "abs_q_delta_k": q_delta_abs_plot,
            "qk": qk_plot,
            "abs_qk": qk_abs_plot,
            "abs_qdelta_over_absqk_qk_ge_1": rel_plot,
            "abs_qdelta_over_mean_absqk": rel_mean_plot,
        },
        "stats": stats_by_name,
    }


def save_unnorm_decomposition_tsv(out_path, pos_list, metrics):
    names = [
        "count_all_num_l2",
        "value_corrected_num_l2",
        "key_corrected_num_l2",
        "reconstructed_num_l2",
        "count_all_full_num_cos",
        "value_corrected_full_num_cos",
        "key_corrected_full_num_cos",
        "reconstructed_full_num_cos",
        "key_num_error_l2",
        "value_num_error_l2",
        "key_delta_l2",
        "value_delta_l2",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("pos\t" + "\t".join(names) + "\n")
        for i, pos in enumerate(pos_list):
            vals = [float(metrics[name][i].item()) for name in names]
            f.write(f"{pos}\t" + "\t".join(f"{v:.8e}" for v in vals) + "\n")


def main():
    set_seed(42)
    args = parse_args()
    if args.merge_metric == "v":
        warnings.warn(
            "--merge-metric v is not the intended setting for this decomposition; "
            "use the default --merge-metric k for the main analysis.",
            stacklevel=2,
        )

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
        compare_tag="compare_error_unnorm",
        include_loss_type=True,
    )

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

    route_mask, belong, count = gen_mask_h2o_with_belong_all(
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
    q_head = layer_ctx.rope_qkv[args.layer]["q"].to(ctx.device)[0][head_idx][:, pos_list, :].float()
    k_head = layer_ctx.rope_qkv[args.layer]["k"].to(ctx.device)[0][head_idx].float()
    v_head = layer_ctx.rope_qkv[args.layer]["v"].to(ctx.device)[0][head_idx].float()

    decomp = decompose_count_all_error_unnorm(
        qk_logits=qk_logits,
        v_head=v_head,
        route_mask=route_mask,
        belong_root=belong_root,
        count=count,
        pos_list=pos_list,
    )

    full_num = decomp["full_num"]
    metrics = {
        "count_all_num_l2": l2_per_pos(decomp["count_all_num"], full_num),
        "value_corrected_num_l2": l2_per_pos(decomp["value_corrected_num"], full_num),
        "key_corrected_num_l2": l2_per_pos(decomp["key_corrected_num"], full_num),
        "reconstructed_num_l2": l2_per_pos(decomp["reconstructed_num"], full_num),
        "count_all_full_num_cos": cosine_per_pos(decomp["count_all_num"], full_num),
        "value_corrected_full_num_cos": cosine_per_pos(decomp["value_corrected_num"], full_num),
        "key_corrected_full_num_cos": cosine_per_pos(decomp["key_corrected_num"], full_num),
        "reconstructed_full_num_cos": cosine_per_pos(decomp["reconstructed_num"], full_num),
        "key_num_error_l2": norm_per_pos(decomp["key_num_error"]),
        "value_num_error_l2": norm_per_pos(decomp["value_num_error"]),
        "key_delta_l2": norm_per_pos(decomp["key_corrected_num"] - decomp["count_all_num"]),
        "value_delta_l2": norm_per_pos(decomp["value_corrected_num"] - decomp["count_all_num"]),
    }

    recon_abs = torch.norm(decomp["reconstructed_num"] - full_num, p=2, dim=-1)
    print("===== Compare-Error-Unnorm Summary =====")
    print(
        f"layer={args.layer}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}, merge_metric={args.merge_metric}"
    )
    print(
        f"mean count_all num_l2={float(metrics['count_all_num_l2'].mean().item()):.8e}, "
        f"mean value_corrected num_l2={float(metrics['value_corrected_num_l2'].mean().item()):.8e}, "
        f"mean key_corrected num_l2={float(metrics['key_corrected_num_l2'].mean().item()):.8e}, "
        f"mean reconstructed num_l2={float(metrics['reconstructed_num_l2'].mean().item()):.8e}"
    )
    print(
        f"mean key_delta_l2={float(metrics['key_delta_l2'].mean().item()):.8e}, "
        f"mean value_delta_l2={float(metrics['value_delta_l2'].mean().item()):.8e}, "
        f"sanity max reconstructed-full_num l2={float(recon_abs.max().item()):.8e}"
    )
    print(
        f"mean count_all full_num_cos={float(metrics['count_all_full_num_cos'].mean().item()):.8e}, "
        f"mean value_corrected full_num_cos={float(metrics['value_corrected_full_num_cos'].mean().item()):.8e}, "
        f"mean key_corrected full_num_cos={float(metrics['key_corrected_full_num_cos'].mean().item()):.8e}, "
        f"mean reconstructed full_num_cos={float(metrics['reconstructed_full_num_cos'].mean().item()):.8e}"
    )

    visualization = visualize_error(
        out_dir=output_dir,
        q_head=q_head,
        k_head=k_head,
        qk_logits=qk_logits,
        belong_root=belong_root,
        pos_list=pos_list,
        head_labels=head_idx,
        dpi=args.plot_dpi,
    )

    per_pos_path = os.path.join(output_dir, "per_pos_error_unnorm_decomposition.tsv")
    save_unnorm_decomposition_tsv(per_pos_path, pos_list, metrics)

    count_vs_key_plot = os.path.join(output_dir, "per_pos_count_all_vs_key_corrected_unnorm.png")
    plot_per_pos_two_lines(
        out_path=count_vs_key_plot,
        pos_list=pos_list,
        y1=metrics["count_all_num_l2"],
        y2=metrics["key_corrected_num_l2"],
        label1="count_all_num_l2",
        label2="key_corrected_num_l2",
        title="Per-Position Numerator L2: Count-All vs Key-Corrected",
        dpi=args.plot_dpi,
    )

    term_plot = os.path.join(output_dir, "per_pos_key_vs_value_delta_unnorm.png")
    plot_per_pos_two_lines(
        out_path=term_plot,
        pos_list=pos_list,
        y1=metrics["key_delta_l2"],
        y2=metrics["value_delta_l2"],
        label1="key_delta_l2",
        label2="value_delta_l2",
        title="Per-Position Numerator Delta Norm: Key vs Value",
        dpi=args.plot_dpi,
    )

    belong_path = os.path.join(output_dir, "belong.pt")
    torch.save(belong.detach().cpu(), belong_path)

    stats = {
        "config": vars(args),
        "layer": int(args.layer),
        "heads": head_idx,
        "pos_list": pos_list,
        "metric_name": "unnormalized_numerator_l2",
        "mean_count_all_num_l2": float(metrics["count_all_num_l2"].mean().item()),
        "mean_value_corrected_num_l2": float(metrics["value_corrected_num_l2"].mean().item()),
        "mean_key_corrected_num_l2": float(metrics["key_corrected_num_l2"].mean().item()),
        "mean_reconstructed_num_l2": float(metrics["reconstructed_num_l2"].mean().item()),
        "mean_count_all_full_num_cos": float(metrics["count_all_full_num_cos"].mean().item()),
        "mean_value_corrected_full_num_cos": float(metrics["value_corrected_full_num_cos"].mean().item()),
        "mean_key_corrected_full_num_cos": float(metrics["key_corrected_full_num_cos"].mean().item()),
        "mean_reconstructed_full_num_cos": float(metrics["reconstructed_full_num_cos"].mean().item()),
        "mean_key_delta_l2": float(metrics["key_delta_l2"].mean().item()),
        "mean_value_delta_l2": float(metrics["value_delta_l2"].mean().item()),
        "mean_key_num_error_l2": float(metrics["key_num_error_l2"].mean().item()),
        "mean_value_num_error_l2": float(metrics["value_num_error_l2"].mean().item()),
        "delta_k_q_delta_k_stats_path": visualization["stats_path"],
        "delta_k_q_delta_k_tensor_path": visualization["tensor_path"],
        "delta_k_q_delta_k_plot_paths": visualization["plots"],
        "delta_k_q_delta_k_stats": visualization["stats"],
        "sanity_mean_reconstructed_full_num_l2": float(recon_abs.mean().item()),
        "sanity_max_reconstructed_full_num_l2": float(recon_abs.max().item()),
        "belong": belong.detach().cpu(),
        "belong_root": belong_root.detach().cpu(),
        "count": count.detach().cpu(),
    }
    for name, value in metrics.items():
        stats[f"{name}_per_pos"] = value.detach().cpu()

    stats_path = os.path.join(output_dir, "compare_error_unnorm_stats.pt")
    torch.save(stats, stats_path)

    print(f"Saved per-pos unnorm decomposition table to: {per_pos_path}")
    print(f"Saved count/key unnorm plot to: {count_vs_key_plot}")
    print(f"Saved key/value unnorm delta plot to: {term_plot}")
    print(f"Saved delta-K/q-delta-K stats to: {visualization['stats_path']}")
    for plot_name, plot_path in visualization["plots"].items():
        print(f"Saved {plot_name} heatmap to: {plot_path}")
    print(f"Saved delta-K/q-delta-K tensors to: {visualization['tensor_path']}")
    print(f"Saved belong tensor to: {belong_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
