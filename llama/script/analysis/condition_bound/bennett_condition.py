"""Stage 1: Bennett condition vs cosh condition (offline, oracle sigma).

Per layer:
  tightness   : total-bound slack err/bound and violations, old vs new
  spearman    : per-cluster condition vs true per-cluster error
  selection   : true hybrid error at matched top-k, ranked by old vs new share
  certificate : certified eps after production-share selection, old vs new
  calibration : eps -> selected-token budget (causal fraction), old vs new
"""

import math

import torch

import common

FRACS = [0.05, 0.1, 0.2]
EPS_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def G_fn(sigma2, delta):
    """Bennett bound on (1/n) sum e^{x_i}: mean 0, var sigma2, x <= delta. float64 in/out."""
    delta = delta.clamp(max=80.0)
    out = torch.ones_like(sigma2)
    ok = (delta > 0) & (sigma2 > 0)
    s2, d = sigma2[ok], delta[ok]
    out[ok] = (s2 * torch.exp(d) + d * d * torch.exp(-s2 / d)) / (s2 + d * d)
    return torch.minimum(out, torch.cosh(delta))


def group_data(q, k_head, v_head, query_pos, block_size, range_delta_fn):
    total = int(query_pos) + 1
    scale = math.sqrt(q.numel())
    qf = q.float()
    vis_k = k_head[:total].float()
    vis_v = v_head[:total].float()
    o_full = (torch.softmax(torch.mv(vis_k, qf) / scale, dim=0).unsqueeze(-1) * vis_v).sum(dim=0)
    b_all = float(torch.norm(vis_v, p=2, dim=-1).max())

    blocks = []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        mem = torch.arange(start, end, device=k_head.device)
        kc = k_head[mem].float()
        vc = v_head[mem].float()
        s = torch.mv(kc, qf) / scale
        s_c = torch.dot(qf, kc.mean(dim=0)) / scale
        blocks.append({
            "size": end - start,
            "z_hat": math.log(end - start) + float(s_c),
            "logZ_true": float(torch.logsumexp(s, dim=0)),
            "num_true": (torch.exp(s - s.max()).unsqueeze(-1) * vc).sum(dim=0) * math.exp(float(s.max())),
            "v_bar": vc.mean(dim=0),
            "delta": float(range_delta_fn(qf, kc, s_c, scale)),
            "sigma2": float(s.var(unbiased=False)),
            "b_c": float(torch.norm(vc, p=2, dim=-1).max()),
        })
    return blocks, o_full, b_all, total


def shares(blocks, b_all, kind):
    p = torch.softmax(torch.tensor([b["z_hat"] for b in blocks], dtype=torch.float64), dim=0)
    delta = torch.tensor([b["delta"] for b in blocks], dtype=torch.float64).clamp(max=80.0)
    b_c = torch.tensor([b["b_c"] for b in blocks], dtype=torch.float64)
    if kind == "cosh":
        F = torch.cosh(delta)
    else:
        F = G_fn(torch.tensor([b["sigma2"] for b in blocks], dtype=torch.float64), delta)
    S = float((p * F).sum())
    share = p * (2 * b_all * (F - 1.0) / S + 2 * b_c * torch.tanh(delta / 2.0))
    return share, p, F, b_c, delta


def hybrid_err(blocks, sel, o_full):
    num, den = 0.0, 0.0
    for blk, s in zip(blocks, sel):
        if s:
            num = num + blk["num_true"]
            den += math.exp(blk["logZ_true"])
        else:
            z = math.exp(blk["z_hat"])
            num = num + z * blk["v_bar"]
            den += z
    return float(torch.norm(num / den - o_full))


def main():
    args = common.parse_args("Bennett vs cosh condition, stage 1")
    ctx = common.load_ctx(args)
    from analysis.condition_block_corr import _pearson, _range_bound_delta, _rankdata

    for layer in args.layers:
        groups = [group_data(q, k, v, qp, args.block_size, _range_bound_delta)
                  for q, k, v, qp in common.layer_groups(ctx, args, layer)]

        slack = {"cosh": [], "bennett": []}
        viol = {"cosh": 0, "bennett": 0}
        sp_x = {"cosh": [], "bennett": []}
        sp_y = []
        sel_err = {k: [[] for _ in FRACS] for k in ("cosh", "bennett")}
        cert = {"cosh": [[] for _ in FRACS], "bennett": [[] for _ in FRACS]}
        budget = {k: [[] for _ in EPS_GRID] for k in ("cosh", "bennett")}

        for blocks, o_full, b_all, total in groups:
            nb = len(blocks)
            true_errs = None
            z_sum = sum(math.exp(b["logZ_true"]) for b in blocks)
            per_kind = {}
            for kind in ("cosh", "bennett"):
                share, p, F, b_c, delta = shares(blocks, b_all, kind)
                per_kind[kind] = (share, p, F, b_c, delta)
                err_tot = hybrid_err(blocks, [False] * nb, o_full)
                bound = float(share.sum())
                slack[kind].append(err_tot / max(bound, 1e-30))
                viol[kind] += err_tot > bound
                sp_x[kind].append(share.float())
                if true_errs is None:
                    true_errs = torch.tensor([
                        float(torch.norm(blk["num_true"] / z_sum - float(p[i]) * blk["v_bar"]))
                        for i, blk in enumerate(blocks)])
                    sp_y.append(true_errs)
                for ei, eps in enumerate(EPS_GRID):
                    sel = (share > eps).tolist()
                    tok = sum(b["size"] for b, s in zip(blocks, sel) if s) + sum(1 for s in sel if not s)
                    budget[kind][ei].append(tok / total)

            share_old = per_kind["cosh"][0]
            for fi, frac in enumerate(FRACS):
                k = int(round(frac * nb))
                for kind in ("cosh", "bennett"):
                    share = per_kind[kind][0]
                    order = torch.argsort(share, descending=True)
                    sel = [False] * nb
                    for j in order[:k].tolist():
                        sel[j] = True
                    sel_err[kind][fi].append(hybrid_err(blocks, sel, o_full))
                # certificates for the SAME (production cosh-share) selection
                order = torch.argsort(share_old, descending=True)
                sel = [False] * nb
                for j in order[:k].tolist():
                    sel[j] = True
                for kind in ("cosh", "bennett"):
                    share, p, F, b_c, delta = per_kind[kind]
                    unsel = [i for i in range(nb) if not sel[i]]
                    T = float(sum(p[i] * (F[i] - 1.0) for i in unsel))
                    c = 2 * b_all * T / (1 + T) + float(
                        sum(2 * p[i] * b_c[i] * torch.tanh(delta[i] / 2.0) for i in unsel))
                    cert[kind][fi].append(c)

        n = len(groups)
        print(f"\n===== layer {layer} ({n} groups, block_size={args.block_size}) =====")
        for kind in ("cosh", "bennett"):
            sl = sorted(slack[kind])
            x = torch.cat(sp_x[kind])
            y = torch.cat(sp_y)
            keep = x > 0
            spear = _pearson(_rankdata(x[keep]), _rankdata(y[keep]))
            print(f"{kind:8s}: bound slack median={sl[n // 2]:.4f} max={sl[-1]:.4f} "
                  f"violations={viol[kind]}/{n} | spearman={spear:.4f}")
        for fi, frac in enumerate(FRACS):
            ec = torch.tensor(sel_err["cosh"][fi]).mean()
            eb = torch.tensor(sel_err["bennett"][fi]).mean()
            cc = torch.tensor(cert["cosh"][fi]).median()
            cb = torch.tensor(cert["bennett"][fi]).median()
            print(f"frac={frac:.2f}: matched-k err cosh={ec:.4f} bennett={eb:.4f} "
                  f"(ratio {eb / ec:.3f}) | cert (same sel) cosh={cc:.3f} bennett={cb:.3f} "
                  f"({cc / cb:.1f}x tighter)")
        print("eps->budget calibration (mean causal token fraction):")
        for ei, eps in enumerate(EPS_GRID):
            bc = torch.tensor(budget["cosh"][ei]).mean()
            bb = torch.tensor(budget["bennett"][ei]).mean()
            print(f"  eps={eps:<6g} cosh={bc:.4f} bennett={bb:.4f}")


if __name__ == "__main__":
    main()
