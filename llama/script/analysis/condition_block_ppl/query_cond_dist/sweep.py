"""Consistency sweep for the query-condition distribution finding.

Loads the model once and, across layers / block sizes / heads, recomputes the
compressed-vs-expanded query summary (see run.py) to test whether the headline
result holds: the queries a cluster compresses point away from k_bar (negative
q.k_bar / low mass), driven by the mass term rather than delta.

Outputs a per-(layer, block, head) TSV and a per-layer consistency plot.
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .run import (
    RESULT_ROOT, _query_positions, compute_cluster_stats, gather_head_qkv, summarize,
)
from ..condition_block import _resolve_block_size
from ...config import set_seed, str_to_torch_dtype
from ...experiment_utils import add_common_compare_args, resolve_head_indices, validate_common_args
from ...online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ...runtime import load_context
from ...sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_compare_args(
        parser, strategy_choices=["block"], default_strategy="block",
        include_loss_type=False, include_plot_dpi=True,
        prefix_mode_default="full_attention", prefix_mode_choices=("full_attention",),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to sweep (default: [--layer]).")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--blocks", type=int, nargs="+", default=None,
                        help="Block sizes to sweep (default: round(1/budget) or --block-size).")
    parser.add_argument("--condition-eps", type=float, default=0.1)
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--sample-heads", type=int, default=8)
    parser.add_argument("--min-queries", type=int, default=32)
    parser.add_argument("--min-compressed", type=int, default=8)
    parser.add_argument("--tag", type=str, default="sweep")
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()
    validate_common_args(
        args, num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )

    all_heads = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    step = max(1, len(all_heads) // args.sample_heads)
    head_idx = all_heads[::step][: args.sample_heads]

    layers = args.layers or [args.layer]
    blocks = args.blocks or [_resolve_block_size(args)]
    eps = float(args.condition_eps)
    query_positions = _query_positions(args)
    pos_list = list(range(0, args.seq_len))
    print(f"model={os.path.basename(args.model)} sample={args.start} seq_len={args.seq_len} "
          f"layers={layers} blocks={blocks} heads={head_idx} eps={eps} "
          f"queries={len(query_positions)}")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    records = []  # one per (layer, block, head-or-all)
    for layer in layers:
        artifacts = capture_layer_artifacts(
            ctx=ctx, layer_idx=layer, pos_list=pos_list,
            model_inputs=model_inputs, layer_to_patch={},
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer, artifacts)
        dev = layer_ctx.device
        q_all, k_all, v_all = gather_head_qkv(layer_ctx.rope_qkv[layer], head_idx, dev)
        vnorm_all = v_all.norm(dim=-1)

        for block in blocks:
            stats = compute_cluster_stats(
                q_all, k_all, vnorm_all, head_idx, query_positions,
                block, eps, args.min_queries, args.min_compressed, keep_tensors=False,
            )
            if not stats:
                continue
            overall = summarize(stats)
            records.append({"layer": layer, "block": block, "head": "all", **overall})
            for h in head_idx:
                hstats = [s for s in stats if s["head"] == int(h)]
                if hstats:
                    records.append({"layer": layer, "block": block, "head": int(h),
                                    **summarize(hstats)})
            o = overall
            print(f"L{layer:>2} B{block:<3} clusters={o['n_clusters']:<5} "
                  f"frac_comp={o['frac_comp']:.2f} | cos(q,k_bar) comp/exp "
                  f"{o['cos_comp_kbar']:+.3f}/{o['cos_exp_kbar']:+.3f} | s "
                  f"{o['s_comp']:+.2f}/{o['s_exp']:+.2f} | delta "
                  f"{o['delta_comp']:.2f}/{o['delta_exp']:.2f} | frac_lower {o['frac_lower']:.2f}")

    model_tag = os.path.basename(args.model.rstrip("/"))
    out_dir = str(RESULT_ROOT / model_tag / f"{args.tag}_sample{args.start}_eps{eps:g}")
    os.makedirs(out_dir, exist_ok=True)
    _write_tsv(os.path.join(out_dir, "consistency.tsv"), records)
    if len(blocks) == 1:
        _plot_layers(records, blocks[0], head_idx,
                     os.path.join(out_dir, "layers.png"), args.plot_dpi)
    print(f"\nSaved to: {out_dir}")


def _write_tsv(path, records):
    cols = ["layer", "block", "head", "n_clusters", "frac_comp", "frac_lower",
            "cos_comp_kbar", "cos_exp_kbar", "s_comp", "s_exp",
            "delta_comp", "delta_exp", "qnorm_comp", "qnorm_exp",
            "u1proj_comp", "u1proj_exp", "cos_comp_qall"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in records:
            f.write("\t".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols
            ) + "\n")


def _plot_layers(records, block, head_idx, path, dpi):
    overall = [r for r in records if r["head"] == "all" and r["block"] == block]
    overall.sort(key=lambda r: r["layer"])
    per_head = [r for r in records if r["head"] != "all" and r["block"] == block]
    lx = [r["layer"] for r in overall]

    panels = [
        ("cos( mean q , k_bar )", "cos_comp_kbar", "cos_exp_kbar"),
        ("mean s = q.k_bar/sqrt(d)", "s_comp", "s_exp"),
        ("mean delta_C", "delta_comp", "delta_exp"),
        ("mean |q|", "qnorm_comp", "qnorm_exp"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, ck, ek) in zip(axes.flat, panels):
        ax.scatter([r["layer"] for r in per_head], [r[ck] for r in per_head],
                   s=16, alpha=0.35, color="#4C72B0")
        ax.scatter([r["layer"] for r in per_head], [r[ek] for r in per_head],
                   s=16, alpha=0.35, color="#C44E52")
        ax.plot(lx, [r[ck] for r in overall], "-o", color="#204060", label="compressed q")
        ax.plot(lx, [r[ek] for r in overall], "-o", color="#6d1f22", label="expanded q")
        ax.set_title(title); ax.set_xlabel("layer"); ax.legend(fontsize=8)
    fig.suptitle(f"query-condition consistency across layers (block={block}); "
                 f"dots=per-head, lines=overall", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved layer consistency plot: {path}")


if __name__ == "__main__":
    main()
