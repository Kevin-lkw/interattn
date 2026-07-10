"""Bound tightness/correlation for B_C / global-B variants of the condition.

Variants per (delta mode, B_C, global B):
  b_unc  = max||v_i||           (production)      b_cen = max||v_i - v_bar||
  b_all  = max||v|| over prefix (production)
  b_star = max_C(||v_bar_C - o_hat|| + 2 B_C tanh(delta_C/2))   (verifiable centered)
  b_orc  = max_C ||u_C - o_hat||                                 (oracle headroom)

Reports Spearman/log-Pearson vs true per-cluster error, summed-bound slack, violations.
"""

import math

import torch

from . import common


def cluster_rows(q, k_head, v_head, query_pos, block_size, range_delta_fn):
    total = int(query_pos) + 1
    scale = math.sqrt(q.numel())
    qf = q.float()
    vis_k = k_head[:total].float()
    vis_v = v_head[:total].float()
    full_alpha = torch.softmax(torch.mv(vis_k, qf) / scale, dim=0)
    b_all = torch.norm(vis_v, p=2, dim=-1).max()

    z_logits, exact_deltas, range_deltas = [], [], []
    b_unc, b_cen, errors = [], [], []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        mem = torch.arange(start, end, device=k_head.device)
        kc = k_head[mem].float()
        vc = v_head[mem].float()
        k_bar = kc.mean(dim=0)
        v_bar = vc.mean(dim=0)
        qk = torch.mv(kc, qf) / scale
        s_c = torch.dot(qf, k_bar) / scale
        exact_deltas.append((qk - s_c).abs().max())
        range_deltas.append(range_delta_fn(qf, kc, s_c, scale))
        z_logits.append(math.log(end - start) + s_c)
        b_unc.append(torch.norm(vc, p=2, dim=-1).max())
        b_cen.append(torch.norm(vc - v_bar, p=2, dim=-1).max())
        o_c = (full_alpha[mem].unsqueeze(-1) * vc).sum(dim=0)
        u_c = (torch.softmax(qk, dim=0).unsqueeze(-1) * vc).sum(dim=0)
        errors.append((o_c, v_bar, u_c))

    p = torch.softmax(torch.stack(z_logits).float(), dim=0)
    v_bars = torch.stack([e[1] for e in errors])
    u_cs = torch.stack([e[2] for e in errors])
    o_hat = (p.unsqueeze(-1) * v_bars).sum(dim=0)
    b_cen_t = torch.stack(b_cen).float()
    range_t = torch.stack(range_deltas).float()
    return {
        "p": p,
        "delta_exact": torch.stack(exact_deltas).float(),
        "delta_range": range_t,
        "b_unc": torch.stack(b_unc).float(),
        "b_cen": b_cen_t,
        "b_all": b_all,
        "b_star": (torch.norm(v_bars - o_hat, p=2, dim=-1)
                   + 2.0 * b_cen_t * torch.tanh(range_t / 2.0)).max(),
        "b_orc": torch.norm(u_cs - o_hat, p=2, dim=-1).max(),
        "err": torch.stack([torch.norm(o_c - p[i] * v_bar, p=2)
                            for i, (o_c, v_bar, _u) in enumerate(errors)]),
    }


VARIANTS = [
    ("delta_range", "b_unc", "b_all"),
    ("delta_range", "b_cen", "b_all"),
    ("delta_range", "b_cen", "b_star"),
    ("delta_range", "b_cen", "b_orc"),
    ("delta_exact", "b_unc", "b_all"),
    ("delta_exact", "b_cen", "b_star"),
    ("delta_exact", "b_cen", "b_orc"),
]


def main():
    args = common.parse_args("Condition-bound variant comparison")
    ctx = common.load_ctx(args)
    from analysis.condition_block_ppl.condition_block_corr import (
        _condition_from_delta,
        _pearson,
        _range_bound_delta,
        _rankdata,
    )

    def spearman(x, y):
        return _pearson(_rankdata(x), _rankdata(y))

    def log_pearson(x, y):
        return _pearson(torch.log10(x.clamp_min(1e-30)), torch.log10(y.clamp_min(1e-30)))

    for layer in args.layers:
        groups = [cluster_rows(q, k, v, qp, args.block_size, _range_bound_delta)
                  for q, k, v, qp in common.layer_groups(ctx, args, layer)]
        print(f"\n===== layer {layer} (groups={len(groups)}, block_size={args.block_size}) =====")
        for delta_key, b_key, b_glob in VARIANTS:
            conds, errs, ratios = [], [], []
            viol = 0
            for g in groups:
                c = _condition_from_delta(g["p"], g[delta_key], g[b_key], g[b_glob]).cpu()
                e = g["err"].cpu()
                conds.append(c)
                errs.append(e)
                r = float(e.sum() / c.sum().clamp_min(1e-30))
                ratios.append(r)
                viol += r > 1.0
            x = torch.cat(conds)
            y = torch.cat(errs)
            keep = x > 0
            ratios.sort()
            n = len(ratios)
            print(f"{delta_key:12s} {b_key:6s} {b_glob:7s} | "
                  f"spearman={spearman(x[keep], y[keep]):.4f} "
                  f"log_pearson={log_pearson(x[keep], y[keep]):.4f} | "
                  f"slack median={ratios[n // 2]:.4f} p90={ratios[int(0.9 * n)]:.4f} "
                  f"max={ratios[-1]:.4f} | violations={viol}/{n}")


if __name__ == "__main__":
    main()
