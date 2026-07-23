"""Per-cluster distribution of the queries that satisfy its compression condition.

Fix a contiguous block cluster C (members [root, root+B), so k_bar_C / key spread
are constant across queries).  Sweep every causal query q that fully sees C and
compute the same coupled condition used for selection:

    s_C     = q . k_bar_C / sqrt(d)          (= segment mean of token scores)
    delta_C = max_i |q.(k_i - k_bar_C)/sqrt(d)|
    p_C     = softmax_C(log|C| + s_C)
    cond_C  = p_C (2B (cosh delta_C - 1) / sum_C' p_C' cosh delta_C'
                   + 2 B_C tanh(delta_C / 2))

C is *compressed* for q when cond_C <= eps.  For each cluster we collect the set
of compressed q, its mean q_bar_comp, and how that mean sits relative to the
cluster geometry (k_bar_C, top centered-key direction) and to the overall query
mean.  The question: do the queries a cluster can compress occupy a characteristic
region of query space, summarised by their mean?
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ..condition import _cluster_condition_rows_and_sanity
from ..condition_block import _build_block_belong_and_mask, _resolve_block_size
from ..per_cluster_condition.scores import condition_scores
from ...config import set_seed, str_to_torch_dtype
from ...experiment_utils import (
    add_common_compare_args,
    resolve_head_indices,
    validate_common_args,
)
from ...online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ...runtime import load_context
from ...sanity import move_model_inputs_to_device

RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "query_cond_dist"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_compare_args(
        parser,
        strategy_choices=["block"],
        default_strategy="block",
        include_loss_type=False,
        include_plot_dpi=True,
        prefix_mode_default="full_attention",
        prefix_mode_choices=("full_attention",),
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--condition-eps", type=float, default=1.0,
                        help="cond_C <= eps means the cluster is compressed for that query.")
    parser.add_argument("--query-start", type=int, default=256,
                        help="First query position (skip the warm-up prefix).")
    parser.add_argument("--num-queries", type=int, default=256,
                        help="Number of query positions, evenly spaced in [query-start, seq-len).")
    parser.add_argument("--sample-heads", type=int, default=4)
    parser.add_argument("--min-queries", type=int, default=24,
                        help="Skip clusters seen by fewer queries than this.")
    parser.add_argument("--min-compressed", type=int, default=6,
                        help="Skip clusters with fewer compressed queries than this.")
    parser.add_argument("--example-clusters", type=int, default=4,
                        help="How many rich clusters to render as PCA scatter panels.")
    return parser.parse_args()


def _cos(a, b, eps=1e-12):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps))


def gather_head_qkv(rope_qkv_layer, head_idx, device):
    """Return (q, k, v) for the requested Q heads, mapping to KV heads under GQA."""
    q_full = rope_qkv_layer["q"].to(device)[0].float()  # [Hq, S, d]
    k_full = rope_qkv_layer["k"].to(device)[0].float()  # [Hkv, S, d]
    v_full = rope_qkv_layer["v"].to(device)[0].float()
    group = q_full.shape[0] // k_full.shape[0]  # 1 for MHA, >1 for GQA
    kv_idx = [int(h) // group for h in head_idx]
    return q_full[list(head_idx)], k_full[kv_idx], v_full[kv_idx]


def _query_positions(args):
    end = args.seq_len
    start = max(0, int(args.query_start))
    if start >= end:
        raise ValueError("--query-start must be < seq-len")
    idx = torch.linspace(start, end - 1, steps=min(args.num_queries, end - start))
    return sorted(set(int(round(x)) for x in idx.tolist()))


def _conditions_for_query(q, k_visible, vnorm_visible, block_size, scale):
    scores = torch.mv(k_visible, q) / scale
    b_all = vnorm_visible.max()
    total = scores.numel()
    n_full = total // block_size
    n_full_tok = n_full * block_size

    s_list, delta_list, bc_list, size_list = [], [], [], []
    if n_full > 0:
        sc = scores[:n_full_tok].view(n_full, block_size)
        vn = vnorm_visible[:n_full_tok].view(n_full, block_size)
        s_full = sc.mean(dim=1)
        d_full = (sc - s_full.unsqueeze(1)).abs().amax(dim=1)
        b_full = vn.amax(dim=1)
        s_list.append(s_full)
        delta_list.append(d_full)
        bc_list.append(b_full)
        size_list.append(torch.full((n_full,), float(block_size), device=q.device))
    if n_full_tok < total:  # partial current block, kept only for normalisation
        sc = scores[n_full_tok:]
        s_p = sc.mean()
        s_list.append(s_p.view(1))
        delta_list.append((sc - s_p).abs().amax().view(1))
        bc_list.append(vnorm_visible[n_full_tok:].max().view(1))
        size_list.append(torch.tensor([float(total - n_full_tok)], device=q.device))

    s_all = torch.cat(s_list)
    delta_all = torch.cat(delta_list)
    bc_all = torch.cat(bc_list)
    size_all = torch.cat(size_list)
    z = torch.log(size_all) + s_all
    p = torch.softmax(z, dim=0)
    cond = condition_scores(p, delta_all, bc_all, b_all)[0]["original"]

    roots = torch.arange(n_full, device=q.device) * block_size
    return roots, cond[:n_full], s_all[:n_full], delta_all[:n_full]


def compute_cluster_stats(q_all, k_all, vnorm_all, head_idx, query_positions,
                          block_size, eps, min_queries, min_compressed,
                          keep_tensors=True):
    """Per-(head, root) stats of the queries a cluster compresses (cond <= eps)."""
    device = q_all.device
    scale = math.sqrt(q_all.shape[-1])
    cluster_stats = []
    for head_ord, head in enumerate(head_idx):
        k_head, vnorm_head, q_head = k_all[head_ord], vnorm_all[head_ord], q_all[head_ord]
        roots_acc, qpos_acc, cond_acc, s_acc, delta_acc = [], [], [], [], []
        for pos in query_positions:
            roots, cond, s, delta = _conditions_for_query(
                q_head[pos], k_head[: pos + 1], vnorm_head[: pos + 1], block_size, scale
            )
            if roots.numel() == 0:
                continue
            roots_acc.append(roots)
            qpos_acc.append(torch.full((roots.numel(),), pos, device=device, dtype=torch.long))
            cond_acc.append(cond)
            s_acc.append(s)
            delta_acc.append(delta)

        roots_acc = torch.cat(roots_acc)
        qpos_acc = torch.cat(qpos_acc)
        cond_acc = torch.cat(cond_acc)
        s_acc = torch.cat(s_acc)
        delta_acc = torch.cat(delta_acc)

        for root in torch.unique(roots_acc).tolist():
            sel = roots_acc == root
            qpos = qpos_acc[sel]
            if qpos.numel() < min_queries:
                continue
            comp = cond_acc[sel] <= eps
            n_exp = int((~comp).sum())
            if int(comp.sum()) < min_compressed:
                continue

            Q = q_head[qpos]
            members = k_head[root : root + block_size]
            k_bar = members.mean(dim=0)
            centered = members - k_bar
            try:
                u1 = torch.linalg.svd(centered, full_matrices=False).Vh[0]
            except Exception:
                u1 = centered[0] / centered[0].norm().clamp_min(1e-12)

            q_all_mean = Q.mean(dim=0)
            q_comp_mean = Q[comp].mean(dim=0)
            q_exp_mean = Q[~comp].mean(dim=0) if n_exp > 0 else torch.zeros_like(q_all_mean)
            qn = Q.norm(dim=-1)
            u1proj = (Q @ u1).abs() / qn.clamp_min(1e-12)
            nan = float("nan")

            row = {
                "head": int(head), "root": int(root),
                "n_all": int(qpos.numel()), "n_comp": int(comp.sum()),
                "frac_comp": float(comp.float().mean()),
                "qbar_all_norm": float(q_all_mean.norm()),
                "qbar_comp_norm": float(q_comp_mean.norm()),
                "qbar_exp_norm": float(q_exp_mean.norm()),
                "cos_comp_kbar": _cos(q_comp_mean, k_bar),
                "cos_exp_kbar": _cos(q_exp_mean, k_bar) if n_exp > 0 else nan,
                "cos_all_kbar": _cos(q_all_mean, k_bar),
                "cos_comp_qall": _cos(q_comp_mean, q_all_mean),
                "cos_exp_qall": _cos(q_exp_mean, q_all_mean) if n_exp > 0 else nan,
                "u1proj_comp": float(u1proj[comp].mean()),
                "u1proj_exp": float(u1proj[~comp].mean()) if n_exp > 0 else nan,
                "s_comp": float(s_acc[sel][comp].mean()),
                "s_exp": float(s_acc[sel][~comp].mean()) if n_exp > 0 else nan,
                "delta_comp": float(delta_acc[sel][comp].mean()),
                "delta_exp": float(delta_acc[sel][~comp].mean()) if n_exp > 0 else nan,
                "qnorm_comp": float(qn[comp].mean()),
                "qnorm_exp": float(qn[~comp].mean()) if n_exp > 0 else nan,
            }
            if keep_tensors:
                row.update({"_Q": Q.detach().cpu(), "_comp": comp.detach().cpu(),
                            "_kbar": k_bar.detach().cpu(), "_u1": u1.detach().cpu()})
            cluster_stats.append(row)
    return cluster_stats


def summarize(stats):
    """Aggregate the per-cluster stats into headline numbers."""
    def col(name):
        return torch.tensor([s[name] for s in stats if not math.isnan(s[name])])

    keys = ["frac_comp", "cos_comp_kbar", "cos_exp_kbar", "s_comp", "s_exp",
            "delta_comp", "delta_exp", "qnorm_comp", "qnorm_exp",
            "u1proj_comp", "u1proj_exp", "cos_comp_qall"]
    out = {"n_clusters": len(stats)}
    for k in keys:
        out[k] = float(col(k).mean()) if len(col(k)) else float("nan")
    pairs = [(s["cos_comp_kbar"], s["cos_exp_kbar"]) for s in stats
             if not math.isnan(s["cos_comp_kbar"]) and not math.isnan(s["cos_exp_kbar"])]
    out["frac_lower"] = (
        sum(1 for c, e in pairs if c < e) / len(pairs) if pairs else float("nan")
    )
    return out


def main():
    set_seed(42)
    args = parse_args()
    block_size = _resolve_block_size(args)
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
        step = max(1, len(head_idx) // args.sample_heads)
        head_idx = head_idx[::step][: args.sample_heads]

    query_positions = _query_positions(args)
    pos_list = list(range(0, args.seq_len))
    print(f"layer={args.layer} block_size={block_size} heads={head_idx} "
          f"queries={len(query_positions)} in [{query_positions[0]},{query_positions[-1]}] "
          f"eps={args.condition_eps}")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx, layer_idx=args.layer, pos_list=pos_list,
        model_inputs=model_inputs, layer_to_patch={},
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)
    device = layer_ctx.device

    q_all, k_all, v_all = gather_head_qkv(layer_ctx.rope_qkv[args.layer], head_idx, device)
    vnorm_all = v_all.norm(dim=-1)
    d = q_all.shape[-1]
    scale = math.sqrt(d)

    # ---- sanity: vectorised condition must match the reference row builder ----
    _run_sanity(q_all, k_all, v_all, vnorm_all, head_ord=0,
                query_pos=query_positions[len(query_positions) // 2],
                block_size=block_size, scale=scale)

    # ---- accumulate per (head, root) the compressed-query statistics ----
    eps = float(args.condition_eps)
    cluster_stats = compute_cluster_stats(
        q_all, k_all, vnorm_all, head_idx, query_positions,
        block_size, eps, args.min_queries, args.min_compressed, keep_tensors=True,
    )

    if not cluster_stats:
        print("No clusters passed the min-queries / min-compressed filters.")
        return

    model_tag = os.path.basename(args.model.rstrip("/"))
    out_dir = args.output_dir or str(
        RESULT_ROOT / model_tag / f"layer{args.layer}_block{block_size}_eps{eps:g}"
    )
    os.makedirs(out_dir, exist_ok=True)
    _write_tsv(os.path.join(out_dir, "cluster_query_dist.tsv"), cluster_stats)
    _print_summary(cluster_stats, eps)
    _plot_aggregate(cluster_stats, os.path.join(out_dir, "aggregate.png"), args.plot_dpi)
    _plot_examples(cluster_stats, os.path.join(out_dir, "example_clusters.png"),
                   args.example_clusters, args.plot_dpi)
    print(f"\nSaved outputs to: {out_dir}")


def _run_sanity(q_all, k_all, v_all, vnorm_all, head_ord, query_pos, block_size, scale):
    device = q_all.device
    belong = (torch.arange(query_pos + 1, device=device) // block_size) * block_size
    ref_rows, _ = _cluster_condition_rows_and_sanity(
        q=q_all[head_ord, query_pos], k_head=k_all[head_ord], v_head=v_all[head_ord],
        row_root=belong, total_available=query_pos + 1, head=0, query_pos=query_pos,
        order="root", delta_mode="exact",
    )
    ref = {r["root"]: r["condition"] for r in ref_rows if r["size"] == block_size}
    roots, cond, _, _ = _conditions_for_query(
        q_all[head_ord, query_pos], k_all[head_ord, : query_pos + 1],
        vnorm_all[head_ord, : query_pos + 1], block_size, scale,
    )
    max_err = 0.0
    for r, c in zip(roots.tolist(), cond.tolist()):
        if r in ref:
            max_err = max(max_err, abs(c - ref[r]) / max(ref[r], 1e-30))
    print(f"[sanity] query_pos={query_pos}: {len(ref)} full clusters, "
          f"max rel cond err vs reference = {max_err:.2e}")
    if max_err > 1e-3:
        raise RuntimeError(f"Vectorised condition disagrees with reference ({max_err:.2e}).")


def _write_tsv(path, stats):
    cols = [k for k in stats[0].keys() if not k.startswith("_")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in stats:
            f.write("\t".join(
                f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in cols
            ) + "\n")


def _print_summary(stats, eps):
    m = summarize(stats)
    print(f"\n===== query-condition distribution ({m['n_clusters']} clusters, eps={eps:g}) =====")
    print(f"mean frac compressed        : {m['frac_comp']:.3f}")
    print("           (compressed q)  (expanded q)")
    for label, cname, ename in [
        ("cos(q_bar, k_bar)   ", "cos_comp_kbar", "cos_exp_kbar"),
        ("|proj q on u1| /|q| ", "u1proj_comp", "u1proj_exp"),
        ("mean s = q.k_bar/sqrt(d)", "s_comp", "s_exp"),
        ("mean delta_C        ", "delta_comp", "delta_exp"),
        ("mean |q|            ", "qnorm_comp", "qnorm_exp"),
    ]:
        print(f"  {label}: {m[cname]:+.4f}      {m[ename]:+.4f}")
    print(f"cos(q_bar_comp, q_bar_all)  : {m['cos_comp_qall']:.3f}")
    print(f"clusters with cos(comp,k_bar) < cos(exp,k_bar): {m['frac_lower']:.2f}")


def _plot_aggregate(stats, path, dpi):
    def col(name):
        return [s[name] for s in stats if not math.isnan(s[name])]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(col("frac_comp"), bins=30, color="#4C72B0")
    ax[0, 0].set_title("fraction of queries compressed per cluster")
    ax[0, 0].set_xlabel("frac_comp")

    ax[0, 1].hist(col("cos_comp_kbar"), bins=30, alpha=0.6, label="compressed q", color="#4C72B0")
    ax[0, 1].hist(col("cos_exp_kbar"), bins=30, alpha=0.6, label="expanded q", color="#C44E52")
    ax[0, 1].set_title("cos( mean q , k_bar_C )")
    ax[0, 1].set_xlabel("cosine"); ax[0, 1].legend()

    ax[1, 0].hist(col("u1proj_comp"), bins=30, alpha=0.6, label="compressed q", color="#4C72B0")
    ax[1, 0].hist(col("u1proj_exp"), bins=30, alpha=0.6, label="expanded q", color="#C44E52")
    ax[1, 0].set_title("|proj of q on top centered-key dir| / |q|")
    ax[1, 0].set_xlabel("alignment with key-spread u1"); ax[1, 0].legend()

    sc = ax[1, 1]
    pairs = [(s["s_exp"], s["s_comp"]) for s in stats
             if not math.isnan(s["s_exp"]) and not math.isnan(s["s_comp"])]
    xe, yc = zip(*pairs) if pairs else ([], [])
    sc.scatter(xe, yc, s=14, alpha=0.6, color="#4C72B0")
    if pairs:
        lim = [min(xe + yc), max(xe + yc)]
        sc.plot(lim, lim, "k--", lw=1)
    sc.set_title("mean s = q.k_bar/sqrt(d): compressed vs expanded")
    sc.set_xlabel("expanded-q mean s"); sc.set_ylabel("compressed-q mean s")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved aggregate plot: {path}")


def _plot_examples(stats, path, n_examples, dpi):
    rich = sorted(stats, key=lambda s: s["n_all"], reverse=True)[:n_examples]
    if not rich:
        return
    cols = min(n_examples, len(rich))
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.0))
    if cols == 1:
        axes = [axes]
    for ax, s in zip(axes, rich):
        Q = s["_Q"].float()
        comp = s["_comp"]
        mean = Q.mean(0)
        _, _, Vh = torch.linalg.svd(Q - mean, full_matrices=False)
        proj = (Q - mean) @ Vh[:2].T
        ax.scatter(proj[~comp, 0], proj[~comp, 1], s=10, alpha=0.5,
                   color="#C44E52", label="expanded")
        ax.scatter(proj[comp, 0], proj[comp, 1], s=10, alpha=0.6,
                   color="#4C72B0", label="compressed")
        cbar = (Q[comp].mean(0) - mean) @ Vh[:2].T
        ebar = (Q[~comp].mean(0) - mean) @ Vh[:2].T
        ax.scatter(*cbar, marker="*", s=260, color="#204060", edgecolor="w", zorder=5)
        ax.scatter(*ebar, marker="*", s=260, color="#6d1f22", edgecolor="w", zorder=5)
        ax.set_title(f"head {s['head']} root {s['root']}\n"
                     f"frac_comp={s['frac_comp']:.2f} n={s['n_all']}")
        ax.set_xlabel("q PC1"); ax.set_ylabel("q PC2")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved example clusters plot: {path}")


if __name__ == "__main__":
    main()
