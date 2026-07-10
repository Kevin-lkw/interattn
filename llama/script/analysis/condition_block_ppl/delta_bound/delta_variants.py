"""Stage 0: how loose is the box delta, and which storable approximation is tighter?

delta_C(q) = max_{i in C} |q (k_i - k_bar_C)| / sqrt(d_k) is the support function of
the symmetrized centered-key hull at q. Every variant below is a valid upper bound
computable from a per-block summary with O(d) query-time work (see README table).

Reports per layer: tightness (median/p90 of delta_hat/delta_oracle, median additive
gap, tanh(delta/2) saturation fraction) and selection quality (Spearman of the share
vs true per-block error; matched-k true hybrid error vs the box baseline).
"""

import math
import torch

from ..condition_bound import common
from ..condition_bound.bennett_condition import hybrid_err

FRACS = [0.05, 0.1, 0.2]
PCA_MS = (1, 2, 4)
EIG_RS = (2, 4)
BOX_CAL = 5.0  # offline-fit median box/oracle ratio; score-only, not a bound
KINDS = ("box", "ball", "diag_ell", "moment", "moment_diag", "eig2", "eig4",
         "moment_orc", "pca1", "pca2", "pca4", "box_cal", "oracle")


def block_deltas(qf, kc, s_c, scale, range_bound_delta):
    """All delta variants for one (query, block): dict kind -> float."""
    kd = kc - kc.mean(dim=0)
    n = kd.shape[0]
    x = torch.mv(kd, qf) / scale  # s_i - mean(s), exactly mean-zero
    out = {"oracle": float(x.abs().max())}
    out["box"] = float(range_bound_delta(qf, kc, s_c, scale))
    out["ball"] = float(qf.norm()) * float(kd.norm(dim=-1).max()) / scale
    w = kd.abs().max(dim=0).values.clamp_min(1e-6)
    rho = float((kd / w).norm(dim=-1).max())
    out["diag_ell"] = rho * float((qf * w).norm()) / scale
    cov = kd.T @ kd / n
    D = torch.diagonal(cov)
    lam = float(torch.linalg.matrix_norm(cov - torch.diag(D), ord=2))
    out["moment"] = math.sqrt(n * float(((qf ** 2 * D).sum() + lam * (qf ** 2).sum())) / (scale ** 2))
    out["moment_diag"] = math.sqrt(n * float((qf ** 2 * D).sum()) / (scale ** 2))  # score-only
    out["moment_orc"] = float(x.norm())  # sqrt(|C|) sigma_oracle = sqrt(sum x_i^2)
    out["box_cal"] = out["box"] / BOX_CAL  # score-only
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals = evals.clamp_min(0.0)
    for r in EIG_RS:
        rr = min(r, evals.numel())
        lam_r = evals[-rr:]
        cq = torch.mv(evecs[:, -rr:].T, qf)
        rem = float(evals[-rr - 1]) if evals.numel() > rr else 0.0
        q_perp2 = max(float((qf ** 2).sum() - (cq ** 2).sum()), 0.0)
        sig2 = (float((lam_r * cq ** 2).sum()) + rem * q_perp2) / (scale ** 2)
        out[f"eig{r}"] = math.sqrt(n * sig2)
    vh = torch.linalg.svd(kd, full_matrices=False).Vh  # (r, d)
    proj = kd @ vh.T                                   # (n, r)
    coef = torch.mv(vh, qf)                            # <q, u_t>
    ext = proj.abs().max(dim=0).values                 # per-direction extent
    for m in PCA_MS:
        mm = min(m, vh.shape[0])
        par = float((coef[:mm].abs() * ext[:mm]).sum())
        q_perp = qf - vh[:mm].T @ coef[:mm]
        r_perp = float((kd - proj[:, :mm] @ vh[:mm]).norm(dim=-1).max())
        out[f"pca{m}"] = (par + float(q_perp.norm()) * r_perp) / scale
    return out


def main():
    args = common.parse_args("Delta approximation variants (stage 0)")
    ctx = common.load_ctx(args)
    from analysis.condition_block_ppl.condition_block_corr import (
        _pearson,
        _range_bound_delta,
        _rankdata,
    )

    for layer in args.layers:
        deltas = {k: [] for k in KINDS}
        sel_err = {k: [[] for _ in FRACS] for k in KINDS}
        sp = {k: [] for k in KINDS}
        sp_y = []
        groups = 0
        for q, k_head, v_head, qpos in common.layer_groups(ctx, args, layer):
            groups += 1
            total = int(qpos) + 1
            scale = math.sqrt(q.numel())
            qf = q.float()
            vis_k = k_head[:total].float()
            vis_v = v_head[:total].float()
            o_full = (torch.softmax(torch.mv(vis_k, qf) / scale, dim=0).unsqueeze(-1) * vis_v).sum(dim=0)
            b_all = float(torch.norm(vis_v, p=2, dim=-1).max())

            blocks = []
            for start in range(0, total, args.block_size):
                end = min(start + args.block_size, total)
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
                    "b_c": float(torch.norm(vc, p=2, dim=-1).max()),
                    "deltas": block_deltas(qf, kc, s_c, scale, _range_bound_delta),
                })

            nb = len(blocks)
            for kind in KINDS:
                deltas[kind].extend(b["deltas"][kind] for b in blocks)
            p = torch.softmax(torch.tensor([b["z_hat"] for b in blocks], dtype=torch.float64), dim=0)
            b_c = torch.tensor([b["b_c"] for b in blocks], dtype=torch.float64)
            z_sum = sum(math.exp(b["logZ_true"]) for b in blocks)
            errs = torch.tensor([
                float(torch.norm(blk["num_true"] / z_sum - float(p[i]) * blk["v_bar"]))
                for i, blk in enumerate(blocks)])
            sp_y.append(errs)
            for kind in KINDS:
                d = torch.tensor([b["deltas"][kind] for b in blocks], dtype=torch.float64).clamp(max=80.0)
                F = torch.cosh(d)
                S = float((p * F).sum())
                share = p * (2 * b_all * (F - 1.0) / S + 2 * b_c * torch.tanh(d / 2.0))
                sp[kind].append(share.float())
                for fi, frac in enumerate(FRACS):
                    k = int(round(frac * nb))
                    order = torch.argsort(share, descending=True)
                    sel = [False] * nb
                    for j in order[:k].tolist():
                        sel[j] = True
                    sel_err[kind][fi].append(hybrid_err(blocks, sel, o_full))

        print(f"\n===== layer {layer} ({groups} groups, block_size={args.block_size}) =====")
        orc = torch.tensor(deltas["oracle"])
        keep = orc > 0.01
        y = torch.cat(sp_y)
        print(f"{keep.sum()}/{len(orc)} blocks with oracle delta > 0.01; "
              f"oracle median delta {orc[keep].median():.2f}, "
              f"tanh-sat(oracle) {(torch.tanh(orc[keep] / 2) > 0.99).float().mean():.2f}")
        for kind in KINDS:
            d = torch.tensor(deltas[kind])
            ratio = d[keep] / orc[keep]
            x = torch.cat(sp[kind])
            good = x > 0
            spear = _pearson(_rankdata(x[good]), _rankdata(y[good]))
            row = (f"{kind:10s}: ratio med={ratio.median():6.2f} p90={ratio.quantile(0.9):6.2f} "
                   f"gap={(d[keep] - orc[keep]).median():5.2f} "
                   f"sat={(torch.tanh(d[keep] / 2) > 0.99).float().mean():.2f} | sp={spear:.4f} |")
            for fi, frac in enumerate(FRACS):
                e = torch.tensor(sel_err[kind][fi]).mean()
                base = torch.tensor(sel_err["box"][fi]).mean()
                row += f" f={frac:.2f}: {e:.4f} ({e / base:.3f}x)"
            print(row)


if __name__ == "__main__":
    main()
