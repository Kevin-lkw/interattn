"""Is the compress/expand boundary a vertical line in q.k_bar, or tilted by delta?

At each query, the real selector keeps the top-k clusters by the full condition
`cond_C` exact (expand) and compresses the rest.  A "mass-only" selector would
instead keep the top-k by `s_C = q.k_bar/sqrt(d)`.  If compression were just
"q.k_bar small", the two selectors would agree.  This script runs both at a
matched per-query budget and cross-tabulates the decisions, so we can see the
clusters the full rule compresses despite large `s` (predicted: small delta) and
the ones it expands despite small `s` (predicted: large delta).

Coordinates: `s_rel = s - s_thresh`, where `s_thresh` is the per-query k-th
largest `s` (the mass-only boundary).  mass-expand == (s_rel > 0).  Plotting
`(s_rel, delta)` coloured by the FULL decision shows whether the real boundary
tilts with delta.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .run import RESULT_ROOT, _conditions_for_query, _query_positions, gather_head_qkv
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
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2],
                        help="Per-query fraction of clusters kept exact (expanded).")
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--sample-heads", type=int, default=8)
    parser.add_argument("--min-clusters", type=int, default=10,
                        help="Skip queries with fewer full clusters than this.")
    parser.add_argument("--plot-budget", type=float, default=0.2)
    return parser.parse_args()


def _per_query_decisions(cond, s, budget):
    """Return (cond_expand, mass_expand, s_rel) boolean/vector for one query."""
    n = cond.numel()
    k = max(1, min(n - 1, int(round(budget * n))))
    cond_expand = torch.zeros(n, dtype=torch.bool, device=cond.device)
    mass_expand = torch.zeros(n, dtype=torch.bool, device=cond.device)
    cond_expand[torch.topk(cond, k).indices] = True
    s_order = torch.topk(s, k)
    mass_expand[s_order.indices] = True
    s_thresh = s_order.values.min()  # k-th largest s = mass boundary
    return cond_expand, mass_expand, s - s_thresh


def main():
    set_seed(42)
    args = parse_args()
    block_size = _resolve_block_size(args)
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

    query_positions = _query_positions(args)
    pos_list = list(range(0, args.seq_len))
    print(f"layer={args.layer} block={block_size} heads={head_idx} "
          f"queries={len(query_positions)} budgets={args.budgets}")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx, layer_idx=args.layer, pos_list=pos_list,
        model_inputs=model_inputs, layer_to_patch={},
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)
    dev = layer_ctx.device
    q_all, k_all, v_all = gather_head_qkv(layer_ctx.rope_qkv[args.layer], head_idx, dev)
    vnorm_all = v_all.norm(dim=-1)
    scale = math.sqrt(q_all.shape[-1])

    # gather per-(q,C) s / delta / cond over all heads and queries (once)
    S, D, C = [], [], []
    for ho in range(len(head_idx)):
        for pos in query_positions:
            roots, cond, s, delta = _conditions_for_query(
                q_all[ho, pos], k_all[ho, : pos + 1], vnorm_all[ho, : pos + 1], block_size, scale
            )
            if roots.numel() < args.min_clusters:
                continue
            S.append(s); D.append(delta); C.append(cond)

    model_tag = os.path.basename(args.model.rstrip("/"))
    out_dir = str(RESULT_ROOT / model_tag / f"boundary_layer{args.layer}_block{block_size}")
    os.makedirs(out_dir, exist_ok=True)

    plot_pack = None
    for budget in args.budgets:
        s_rel_all, delta_all, cond_exp_all, mass_exp_all = [], [], [], []
        for s, delta, cond in zip(S, D, C):
            ce, me, srel = _per_query_decisions(cond, s, budget)
            s_rel_all.append(srel); delta_all.append(delta)
            cond_exp_all.append(ce); mass_exp_all.append(me)
        s_rel = torch.cat(s_rel_all); delta = torch.cat(delta_all)
        ce = torch.cat(cond_exp_all); me = torch.cat(mass_exp_all)
        _report(budget, s_rel, delta, ce, me)
        if abs(budget - args.plot_budget) < 1e-9:
            plot_pack = (budget, s_rel.cpu(), delta.cpu(), ce.cpu(), me.cpu())

    if plot_pack is not None:
        _plot(plot_pack, os.path.join(out_dir, f"boundary_b{plot_pack[0]:g}.png"), args.plot_dpi)
    print(f"\nSaved to: {out_dir}")


def _report(budget, s_rel, delta, ce, me):
    total = ce.numel()
    n_exp = int(ce.sum())
    # 4 cells (cond decision x mass decision)
    agree_exp = ce & me
    agree_comp = (~ce) & (~me)
    only_cond_comp = (~ce) & me      # mass would EXPAND (high s), full COMPRESSES  <- user's set
    only_cond_exp = ce & (~me)       # mass would COMPRESS (low s), full EXPANDS
    disagree = only_cond_comp | only_cond_exp

    def stat(mask):
        if int(mask.sum()) == 0:
            return (0, float("nan"), float("nan"))
        return (int(mask.sum()), float(delta[mask].mean()), float(s_rel[mask].mean()))

    print(f"\n===== budget={budget:g}  ({total} pairs, {n_exp} expanded by full cond) =====")
    print(f"decision disagreement (cond vs mass-only q.k_bar): "
          f"{int(disagree.sum())}/{total} = {float(disagree.float().mean()):.3f}  "
          f"(Jaccard on expand set: {_jaccard(ce, me):.3f})")
    print(f"{'cell':<34}{'count':>8}{'mean delta':>12}{'mean s_rel':>12}")
    for name, mask in [
        ("agree: both expand", agree_exp),
        ("agree: both compress", agree_comp),
        ("full COMPRESS but s high (mass exp)", only_cond_comp),
        ("full EXPAND but s low (mass comp)", only_cond_exp),
    ]:
        n, md, ms = stat(mask)
        print(f"{name:<34}{n:>8}{md:>12.3f}{ms:>12.3f}")
    # direct test of the user's hypothesis
    comp = ~ce
    hi_s_comp = comp & (s_rel > 0)  # compressed despite being above the mass boundary
    print(f"\nUser's set — compressed AND above mass boundary (large q.k_bar, compressed): "
          f"{int(hi_s_comp.sum())} = {float((hi_s_comp.float().sum() / comp.float().sum())):.3f} of all compressed")
    if int(hi_s_comp.sum()) and int(comp.sum()):
        print(f"  their mean delta {float(delta[hi_s_comp].mean()):.3f} "
              f"vs all-compressed mean delta {float(delta[comp].mean()):.3f} "
              f"vs expanded mean delta {float(delta[ce].mean()):.3f}")


def _jaccard(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else float("nan")


def _plot(pack, path, dpi):
    budget, s_rel, delta, ce, me = pack
    # subsample for a legible scatter
    n = s_rel.numel()
    idx = torch.randperm(n)[: min(20000, n)] if n > 20000 else torch.arange(n)
    x, y, c = s_rel[idx], delta[idx], ce[idx]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].scatter(x[~c], y[~c], s=5, alpha=0.25, color="#4C72B0", label="full compresses")
    ax[0].scatter(x[c], y[c], s=5, alpha=0.25, color="#C44E52", label="full expands")
    ax[0].axvline(0.0, color="k", ls="--", lw=1.2, label="mass-only boundary (s_rel=0)")
    ax[0].set_xlabel("s_rel = q.k_bar/sqrt(d) - per-query mass threshold")
    ax[0].set_ylabel("delta_C")
    ax[0].set_title(f"decision in (s_rel, delta), budget={budget:g}")
    lim = float(torch.quantile(s_rel.abs(), 0.99))
    ax[0].set_xlim(-lim, lim); ax[0].legend(fontsize=8, markerscale=2)

    # fraction expanded by full cond, binned over (s_rel, delta) — shows the tilt
    xb = torch.clamp(x, -lim, lim)
    H_exp, xe, ye = _hist2d(xb, y, c.float(), bins=40)
    H_tot, _, _ = _hist2d(xb, y, torch.ones_like(c.float()), bins=40, edges=(xe, ye))
    frac = (H_exp / H_tot.clamp_min(1)).T
    im = ax[1].imshow(frac, origin="lower", aspect="auto",
                      extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="coolwarm", vmin=0, vmax=1)
    ax[1].axvline(0.0, color="k", ls="--", lw=1.2)
    ax[1].set_xlabel("s_rel"); ax[1].set_ylabel("delta_C")
    ax[1].set_title("P(full expands) per (s_rel, delta) bin")
    fig.colorbar(im, ax=ax[1], label="P(expand)")
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)
    print(f"saved boundary plot: {path}")


def _hist2d(x, y, w, bins=40, edges=None):
    if edges is None:
        xe = torch.linspace(float(x.min()), float(x.max()), bins + 1)
        ye = torch.linspace(float(y.min()), float(y.max()), bins + 1)
    else:
        xe, ye = edges
    xi = torch.bucketize(x, xe[1:-1])
    yi = torch.bucketize(y, ye[1:-1])
    H = torch.zeros(bins, bins)
    H.index_put_((xi, yi), w, accumulate=True)
    return H, xe, ye


if __name__ == "__main__":
    main()
