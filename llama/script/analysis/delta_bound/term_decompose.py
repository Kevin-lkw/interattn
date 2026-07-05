"""Which term of the share does the ranking work, and how does delta enter each?

Per block: t1 = p_hat 2B(cosh d - 1)/S_d (mass term, ~ 2B softmax(log p_hat + d)),
           t2 = p_hat 2B_C tanh(d/2)     (value term, = 2 p_hat B_C when saturated).
For delta in {box, oracle}: Spearman vs true per-block error and matched-k true hybrid
error for rankings by t1 only, t2 only, and the full share; plus the term-1 fraction of
the share, overall and among the top-10% selected blocks.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "condition_bound")))
import common
from bennett_condition import hybrid_err

FRACS = [0.05, 0.1, 0.2]
RANKS = [(dk, comp) for dk in ("box", "oracle") for comp in ("full", "t1", "t2")]


def main():
    args = common.parse_args("Share term decomposition (box vs oracle delta)")
    ctx = common.load_ctx(args)
    from analysis.condition_block_corr import _pearson, _range_bound_delta, _rankdata

    for layer in args.layers:
        sel_err = {r: [[] for _ in FRACS] for r in RANKS}
        sp = {r: [] for r in RANKS}
        sp_y = []
        t1f_all, t1f_sel = [], []
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
                x = s - s.mean()
                blocks.append({
                    "size": end - start,
                    "z_hat": math.log(end - start) + float(s_c),
                    "logZ_true": float(torch.logsumexp(s, dim=0)),
                    "num_true": (torch.exp(s - s.max()).unsqueeze(-1) * vc).sum(dim=0) * math.exp(float(s.max())),
                    "v_bar": vc.mean(dim=0),
                    "b_c": float(torch.norm(vc, p=2, dim=-1).max()),
                    "d_box": float(_range_bound_delta(qf, kc, s_c, scale)),
                    "d_orc": float(x.abs().max()),
                })

            nb = len(blocks)
            p = torch.softmax(torch.tensor([b["z_hat"] for b in blocks], dtype=torch.float64), dim=0)
            b_c = torch.tensor([b["b_c"] for b in blocks], dtype=torch.float64)
            z_sum = sum(math.exp(b["logZ_true"]) for b in blocks)
            errs = torch.tensor([
                float(torch.norm(blk["num_true"] / z_sum - float(p[i]) * blk["v_bar"]))
                for i, blk in enumerate(blocks)])
            sp_y.append(errs)
            for dk, key in (("box", "d_box"), ("oracle", "d_orc")):
                d = torch.tensor([b[key] for b in blocks], dtype=torch.float64).clamp(max=80.0)
                F = torch.cosh(d)
                S = float((p * F).sum())
                t1 = p * 2 * b_all * (F - 1.0) / S
                t2 = p * 2 * b_c * torch.tanh(d / 2.0)
                comps = {"full": t1 + t2, "t1": t1, "t2": t2}
                if dk == "box":
                    frac = (t1 / (t1 + t2).clamp_min(1e-30)).float()
                    t1f_all.append(frac)
                    top = torch.argsort(comps["full"], descending=True)[:max(int(round(0.1 * nb)), 1)]
                    t1f_sel.append(frac[top])
                for comp, x in comps.items():
                    sp[(dk, comp)].append(x.float())
                    for fi, fr in enumerate(FRACS):
                        k = int(round(fr * nb))
                        order = torch.argsort(x, descending=True)
                        sel = [False] * nb
                        for j in order[:k].tolist():
                            sel[j] = True
                        sel_err[(dk, comp)][fi].append(hybrid_err(blocks, sel, o_full))

        print(f"\n===== layer {layer} ({groups} groups, block_size={args.block_size}) =====")
        fa, fs = torch.cat(t1f_all), torch.cat(t1f_sel)
        print(f"term1 share fraction (box): median {fa.median():.3f} all blocks, "
              f"{fs.median():.3f} among top-10% selected; "
              f"term1-dominant blocks {(fa > 0.5).float().mean():.3f} all, {(fs > 0.5).float().mean():.3f} selected")
        y = torch.cat(sp_y)
        for r in RANKS:
            x = torch.cat(sp[r])
            good = x > 0
            spear = _pearson(_rankdata(x[good]), _rankdata(y[good]))
            row = f"{r[0]:6s}-{r[1]:4s}: sp={spear:.4f} |"
            for fi, fr in enumerate(FRACS):
                e = torch.tensor(sel_err[r][fi]).mean()
                base = torch.tensor(sel_err[("box", "full")][fi]).mean()
                row += f" f={fr:.2f}: {e:.4f} ({e / base:.3f}x)"
            print(row)


if __name__ == "__main__":
    main()
