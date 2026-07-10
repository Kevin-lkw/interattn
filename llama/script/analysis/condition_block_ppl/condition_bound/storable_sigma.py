"""Stage 2: storable sigma vs oracle sigma for the Bennett selection score.

Sigma variants per (query, block):
  oracle  : Var_i(s_i) exactly
  diag    : sum_j q_j^2 D_j / d_k          (storable; NOT an upper bound)
  diag+lam: (sum_j q_j^2 D_j + lam ||q||^2)/d_k  (storable upper bound; lam = ||cov - diag||_2)

Reports matched-k true hybrid selection error (vs cosh share and oracle-Bennett share)
and Spearman, at the generation block size (default budget 0.03125 -> block 32).
"""

import math

import torch

from . import common
from .bennett_condition import G_fn, hybrid_err

FRACS = [0.05, 0.1, 0.2]


def main():
    args = common.parse_args("Storable-sigma Bennett selection check")
    ctx = common.load_ctx(args)
    from analysis.condition_block_ppl.condition_block_corr import (
        _pearson,
        _range_bound_delta,
        _rankdata,
    )

    for layer in args.layers:
        kinds = ("cosh", "oracle", "diag", "diag_lam")
        sel_err = {k: [[] for _ in FRACS] for k in kinds}
        sp = {k: [] for k in kinds}
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
            sig2 = {"oracle": [], "diag": [], "diag_lam": []}
            for start in range(0, total, args.block_size):
                end = min(start + args.block_size, total)
                mem = torch.arange(start, end, device=k_head.device)
                kc = k_head[mem].float()
                vc = v_head[mem].float()
                s = torch.mv(kc, qf) / scale
                s_c = torch.dot(qf, kc.mean(dim=0)) / scale
                kd = kc - kc.mean(dim=0)
                cov = kd.T @ kd / kd.shape[0]
                D = torch.diagonal(cov)
                lam = float(torch.linalg.matrix_norm(cov - torch.diag(D), ord=2))
                qsq = qf ** 2
                sig2["oracle"].append(float(s.var(unbiased=False)))
                sig2["diag"].append(float((qsq * D).sum() / (scale ** 2)))
                sig2["diag_lam"].append(float(((qsq * D).sum() + lam * (qf ** 2).sum()) / (scale ** 2)))
                blocks.append({
                    "size": end - start,
                    "z_hat": math.log(end - start) + float(s_c),
                    "logZ_true": float(torch.logsumexp(s, dim=0)),
                    "num_true": (torch.exp(s - s.max()).unsqueeze(-1) * vc).sum(dim=0) * math.exp(float(s.max())),
                    "v_bar": vc.mean(dim=0),
                    "delta": float(_range_bound_delta(qf, kc, s_c, scale)),
                    "b_c": float(torch.norm(vc, p=2, dim=-1).max()),
                })

            nb = len(blocks)
            p = torch.softmax(torch.tensor([b["z_hat"] for b in blocks], dtype=torch.float64), dim=0)
            delta = torch.tensor([b["delta"] for b in blocks], dtype=torch.float64).clamp(max=80.0)
            b_c = torch.tensor([b["b_c"] for b in blocks], dtype=torch.float64)
            tanh_h = torch.tanh(delta / 2.0)
            shares = {}
            for kind in kinds:
                if kind == "cosh":
                    F = torch.cosh(delta)
                else:
                    F = G_fn(torch.tensor(sig2[kind], dtype=torch.float64), delta)
                S = float((p * F).sum())
                shares[kind] = p * (2 * b_all * (F - 1.0) / S + 2 * b_c * tanh_h)

            z_sum = sum(math.exp(b["logZ_true"]) for b in blocks)
            errs = torch.tensor([
                float(torch.norm(blk["num_true"] / z_sum - float(p[i]) * blk["v_bar"]))
                for i, blk in enumerate(blocks)])
            sp_y.append(errs)
            for kind in kinds:
                sp[kind].append(shares[kind].float())
                for fi, frac in enumerate(FRACS):
                    k = int(round(frac * nb))
                    order = torch.argsort(shares[kind], descending=True)
                    sel = [False] * nb
                    for j in order[:k].tolist():
                        sel[j] = True
                    sel_err[kind][fi].append(hybrid_err(blocks, sel, o_full))

        print(f"\n===== layer {layer} ({groups} groups, block_size={args.block_size}) =====")
        y = torch.cat(sp_y)
        for kind in kinds:
            x = torch.cat(sp[kind])
            keep = x > 0
            spear = _pearson(_rankdata(x[keep]), _rankdata(y[keep]))
            row = f"{kind:9s}: spearman={spear:.4f} |"
            for fi, frac in enumerate(FRACS):
                e = torch.tensor(sel_err[kind][fi]).mean()
                base = torch.tensor(sel_err["cosh"][fi]).mean()
                row += f" f={frac:.2f}: err={e:.4f} ({e / base:.3f}x cosh)"
            print(row)


if __name__ == "__main__":
    main()
