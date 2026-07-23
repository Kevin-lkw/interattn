"""Offline: is avg-K / avg-V the best cluster representative for the true error?

For a fixed budget b (fraction of clusters kept exact per query), we select the
exact set by the real condition, compress the rest, and measure the true output
error ||o_hat - o|| under different representatives for the compressed clusters.

Value representatives (mass fixed at k_bar):
  vbar     uniform mean (baseline / current method)
  vnorm    norm-weighted mean            (cheap control)
  v_all    sum_i wbar_i^all  v_i         (attention-weighted over ALL queries)
  v_qw     sum_i wbar_i^comp v_i         (attention-weighted over COMPRESSING queries)
  oracle_u u_C(q) per query              (value ceiling: only mass error remains)
Key representatives (value fixed at v_bar):
  k_qw     k* = sum_i wbar_i^comp k_i    (attention-weighted key; loosens the bound)
  both_qw  (k_qw, v_qw)
  oracle_m Z_C true mass, v_bar          (mass ceiling: only value error remains)

wbar_i are query-averaged within-cluster softmax weights (Prop. 1 in derivation.tex):
the storable optimum is the conditional mean v* = E_q[u_C].  Uses the exact-mass /
exact-value identity  sum_{i in C} e^{s_i} v_i = Z_C u_C  to assemble the hybrid output
cheaply.  Reuses query_cond_dist / per_cluster_condition; touches no existing code.
"""

import argparse
import math
import os
from pathlib import Path

import torch

from ..per_cluster_condition.scores import condition_scores
from ..query_cond_dist.run import _query_positions, gather_head_qkv
from ...config import set_seed, str_to_torch_dtype
from ...experiment_utils import add_common_compare_args, resolve_head_indices, validate_common_args
from ..condition_block import _resolve_block_size
from ...online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from ...runtime import load_context
from ...sanity import move_model_inputs_to_device

RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "representative_optimality"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_compare_args(
        parser, strategy_choices=["block"], default_strategy="block",
        include_loss_type=False, include_plot_dpi=True,
        prefix_mode_default="full_attention", prefix_mode_choices=("full_attention",),
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--select-budgets", type=float, nargs="+", default=[0.0, 0.1, 0.2],
                        help="Fraction of clusters kept exact per query; 0 = pure approximation.")
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--sample-heads", type=int, default=8)
    parser.add_argument("--min-compress-count", type=int, default=8,
                        help="Clusters compressed by fewer queries fall back to vbar for v_qw.")
    return parser.parse_args()


def _per_query(q, k_vis, v_vis, block_size, scale):
    """Per full block: true mass Z, true value u, mean stats, cond; plus partial block."""
    scores = torch.mv(k_vis, q) / scale
    vnorm = v_vis.norm(dim=-1)
    b_all = vnorm.max()
    T = scores.numel()
    d = v_vis.shape[-1]
    n_full = T // block_size
    nft = n_full * block_size

    out = {"n_full": n_full, "b_all": b_all}
    if n_full > 0:
        sc = scores[:nft].view(n_full, block_size)
        vblk = v_vis[:nft].view(n_full, block_size, d)
        vnblk = vnorm[:nft].view(n_full, block_size)
        e = torch.exp(sc - sc.max(dim=1, keepdim=True).values)  # stable within-block
        w = e / e.sum(dim=1, keepdim=True)                      # softmax weights
        Z = torch.logsumexp(sc, dim=1).exp()                    # true mass
        u = (w.unsqueeze(-1) * vblk).sum(dim=1)                 # true value
        s = sc.mean(dim=1)
        out.update(dict(
            s=s, Zhat=block_size * torch.exp(s), Z=Z, u=u, w=w,
            delta=(sc - s.unsqueeze(1)).abs().amax(dim=1),
            b_c=vnblk.amax(dim=1),
        ))
    # partial current block (always kept exact); also feeds the softmax normaliser
    if nft < T:
        sp = scores[nft:]
        ep = torch.exp(sp - sp.max())
        Zp = torch.logsumexp(sp, dim=0).exp()
        up = (ep / ep.sum()) @ v_vis[nft:]
        out["partial"] = (Zp, up, sp.mean(), (sp - sp.mean()).abs().max(), vnorm[nft:].max(),
                          float(T - nft))
    else:
        out["partial"] = None
    return out


def _cond_and_select(pq, budget, block_size):
    """Return (compressed_mask over full blocks, p_hat over full blocks)."""
    n = pq["n_full"]
    s = pq["s"]; delta = pq["delta"]; b_c = pq["b_c"]
    s_all = [s]; d_all = [delta]; bc_all = [b_c]
    size = [torch.full((n,), float(block_size), device=s.device)]
    if pq["partial"] is not None:
        Zp, up, sp, dp, bcp, szp = pq["partial"]
        s_all.append(sp.view(1)); d_all.append(dp.view(1)); bc_all.append(bcp.view(1))
        size.append(torch.tensor([szp], device=s.device))
    s_all = torch.cat(s_all); d_all = torch.cat(d_all)
    bc_all = torch.cat(bc_all); size = torch.cat(size)
    z = torch.log(size) + s_all
    p = torch.softmax(z, dim=0)
    cond = condition_scores(p, d_all, bc_all, pq["b_all"])[0]["original"][:n]
    k = max(0, min(n, int(round(budget * n))))
    compressed = torch.ones(n, dtype=torch.bool, device=s.device)
    if k > 0:
        compressed[torch.topk(cond, k).indices] = False
    return compressed, p[:n]


def _true_output(pq):
    num = torch.zeros_like(pq["u"][0]) if pq["n_full"] else None
    den = torch.zeros((), device=pq["b_all"].device)
    if pq["n_full"]:
        num = (pq["Z"].unsqueeze(-1) * pq["u"]).sum(dim=0)
        den = pq["Z"].sum()
    if pq["partial"] is not None:
        Zp, up = pq["partial"][0], pq["partial"][1]
        num = up * Zp if num is None else num + Zp * up
        den = den + Zp
    return num / den.clamp_min(1e-30)


def _hybrid_output(pq, compressed, mass_rep, value_rep):
    """mass_rep/value_rep: [n_full] and [n_full,d] used for compressed clusters."""
    Z, u = pq["Z"], pq["u"]
    exp_m = torch.where(compressed, mass_rep, Z)            # mass per full block
    exp_val = torch.where(compressed.unsqueeze(-1), value_rep, u)
    num = (exp_m.unsqueeze(-1) * exp_val).sum(dim=0)
    den = exp_m.sum()
    if pq["partial"] is not None:
        Zp, up = pq["partial"][0], pq["partial"][1]
        num = num + Zp * up
        den = den + Zp
    return num / den.clamp_min(1e-30)


def main():
    set_seed(42)
    args = parse_args()
    block_size = _resolve_block_size(args)
    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()
    validate_common_args(args, num_layers=ctx.model_config.num_hidden_layers,
                         num_heads=ctx.model_config.num_attention_heads)
    all_heads = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    step = max(1, len(all_heads) // args.sample_heads)
    head_idx = all_heads[::step][: args.sample_heads]

    query_positions = _query_positions(args)
    pos_list = list(range(0, args.seq_len))
    print(f"layer={args.layer} block={block_size} heads={head_idx} "
          f"queries={len(query_positions)} budgets={args.select_budgets}")

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(ctx=ctx, layer_idx=args.layer, pos_list=pos_list,
                                        model_inputs=model_inputs, layer_to_patch={})
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)
    dev = layer_ctx.device
    q_all, k_all, v_all = gather_head_qkv(layer_ctx.rope_qkv[args.layer], head_idx, dev)
    scale = math.sqrt(q_all.shape[-1])

    variants = ["vbar", "vnorm", "v_all", "v_qw", "v_qmean", "oracle_u", "k_qw",
                "both_qw", "oracle_m"]
    # error accumulators: {budget: {variant: [sum_rel, sum_abs]}}, plus count and ||o||
    acc = {b: {v: [0.0, 0.0] for v in variants} for b in args.select_budgets}
    acc_n = {b: 0 for b in args.select_budgets}

    for ho in range(len(head_idx)):
        # cache per-query quantities once for this head (budget-independent)
        # calibration/eval split: even-indexed queries build reps, odd ones evaluate
        cache = []
        for pos in query_positions:
            pq = _per_query(q_all[ho, pos], k_all[ho, : pos + 1], v_all[ho, : pos + 1],
                            block_size, scale)
            if pq["n_full"] == 0:
                continue
            cache.append((pos, pq, len(cache) % 2 == 0))

        n_roots = args.seq_len // block_size
        v_static = v_all[ho, : n_roots * block_size].view(n_roots, block_size, -1)
        k_static = k_all[ho, : n_roots * block_size].view(n_roots, block_size, -1)
        vbar_root = v_static.mean(dim=1)                                   # [n_roots,d]
        vn = v_static.norm(dim=-1)
        vnorm_root = (vn.unsqueeze(-1) * v_static).sum(1) / vn.sum(1, keepdim=True)

        # all-query weight average from calibration queries (budget-independent)
        wsum_all = torch.zeros(n_roots, block_size, device=dev)
        wcnt_all = torch.zeros(n_roots, device=dev)
        for _, pq, is_cal in cache:
            if not is_cal:
                continue
            nf = pq["n_full"]
            wsum_all[:nf] += pq["w"]; wcnt_all[:nf] += 1
        wbar_all = wsum_all / wcnt_all.clamp_min(1).unsqueeze(-1)
        v_all_rep = (wbar_all.unsqueeze(-1) * v_static).sum(1)

        # cheap storable proxy: weights from the mean calibration query direction q_bar
        q_cal = torch.stack([q_all[ho, pos] for pos, _, is_cal in cache if is_cal])
        q_bar = q_cal.mean(dim=0)
        w_qm = torch.softmax((k_static @ q_bar) / scale, dim=1)            # [n_roots,block]
        v_qmean_rep = (w_qm.unsqueeze(-1) * v_static).sum(1)

        for budget in args.select_budgets:
            # pass 1: compress-distribution weights from CALIBRATION queries at this budget
            wsum = torch.zeros(n_roots, block_size, device=dev)
            wcnt = torch.zeros(n_roots, device=dev)
            for _, pq, is_cal in cache:
                if not is_cal:
                    continue
                nf = pq["n_full"]
                comp, _ = _cond_and_select(pq, budget, block_size)
                wsum[:nf] += torch.where(comp.unsqueeze(-1), pq["w"], torch.zeros_like(pq["w"]))
                wcnt[:nf] += comp.float()
            enough = wcnt >= args.min_compress_count
            wbar = torch.where(enough.unsqueeze(-1), wsum / wcnt.clamp_min(1).unsqueeze(-1),
                               torch.full_like(wsum, 1.0 / block_size))
            v_qw_rep = (wbar.unsqueeze(-1) * v_static).sum(1)              # [n_roots,d]
            k_qw_rep = (wbar.unsqueeze(-1) * k_static).sum(1)             # [n_roots,d]

            # pass 2: true error on EVALUATION queries per variant
            for pos, pq, is_cal in cache:
                if is_cal:
                    continue
                nf = pq["n_full"]
                comp, _ = _cond_and_select(pq, budget, block_size)
                o = _true_output(pq)
                onorm = o.norm().clamp_min(1e-30)
                s_kqw = (k_qw_rep[:nf] * q_all[ho, pos]).sum(-1) / scale
                zhat_kqw = block_size * torch.exp(s_kqw)
                reps = {
                    "vbar":     (pq["Zhat"], vbar_root[:nf]),
                    "vnorm":    (pq["Zhat"], vnorm_root[:nf]),
                    "v_all":    (pq["Zhat"], v_all_rep[:nf]),
                    "v_qw":     (pq["Zhat"], v_qw_rep[:nf]),
                    "v_qmean":  (pq["Zhat"], v_qmean_rep[:nf]),
                    "oracle_u": (pq["Zhat"], pq["u"]),
                    "k_qw":     (zhat_kqw, vbar_root[:nf]),
                    "both_qw":  (zhat_kqw, v_qw_rep[:nf]),
                    "oracle_m": (pq["Z"], vbar_root[:nf]),
                }
                for name in variants:
                    mass_rep, value_rep = reps[name]
                    oh = _hybrid_output(pq, comp, mass_rep, value_rep)
                    err = (oh - o).norm()
                    acc[budget][name][0] += float(err / onorm)
                    acc[budget][name][1] += float(err)
                acc_n[budget] += 1

    model_tag = os.path.basename(args.model.rstrip("/"))
    out_dir = str(RESULT_ROOT / model_tag / f"layer{args.layer}_block{block_size}")
    os.makedirs(out_dir, exist_ok=True)
    _report_and_save(acc, acc_n, variants, os.path.join(out_dir, "offline.tsv"))
    print(f"\nSaved to: {out_dir}")


def _report_and_save(acc, acc_n, variants, path):
    lines = ["budget\tn\t" + "\t".join(variants)]
    print(f"\n{'budget':>7} {'n':>6}  " + "  ".join(f"{v:>9}" for v in variants))
    print("rel error ||o_hat-o||/||o|| (mean over queries*heads); lower is better")
    for b in sorted(acc):
        n = max(1, acc_n[b])
        rels = {v: acc[b][v][0] / n for v in variants}
        print(f"{b:>7g} {n:>6}  " + "  ".join(f"{rels[v]:>9.4f}" for v in variants))
        lines.append(f"{b:g}\t{n}\t" + "\t".join(f"{rels[v]:.6g}" for v in variants))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
