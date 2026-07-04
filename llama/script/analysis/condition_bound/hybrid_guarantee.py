"""Verify the per-cluster guarantee and hybrid (selected-exact) certificate.

Per (head, query): rank blocks, select top-k, compute the TRUE hybrid output
(selected blocks exact, unselected use |C|e^{s_C} mass and v_bar, shared softmax),
and compare against
  cert_tight = 2B T/(1+T) + sum_unsel 2 p_hat B_C tanh(delta/2),  T = sum_unsel p_hat(cosh-1)
  cert_loose = sum_unsel p_hat [2B(cosh-1) + 2 B_C tanh(delta/2)]
Also compares selection quality (true error at matched k) for ranking by the certified
(unnormalized) score vs the production normalized share, and checks the literal
per-cluster guarantee p_hat[B max(cosh-1, (S-1)/S) + 2 B_C tanh(delta/2)].
"""

import math

import torch

import common

FRACS = [0.0, 0.05, 0.1, 0.2, 0.4]


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
            "z_hat": math.log(end - start) + float(s_c),
            "logZ_true": float(torch.logsumexp(s, dim=0)),
            "num_true": (torch.exp(s - s.max()).unsqueeze(-1) * vc).sum(dim=0) * math.exp(float(s.max())),
            "v_bar": vc.mean(dim=0),
            "delta": float(range_delta_fn(qf, kc, s_c, scale)),
            "b_c": float(torch.norm(vc, p=2, dim=-1).max()),
        })
    return blocks, o_full, b_all


def hybrid_error(blocks, sel_mask, o_full):
    num = 0.0
    den = 0.0
    for blk, sel in zip(blocks, sel_mask):
        if sel:
            num = num + blk["num_true"]
            den += math.exp(blk["logZ_true"])
        else:
            z = math.exp(blk["z_hat"])
            num = num + z * blk["v_bar"]
            den += z
    return float(torch.norm(num / den - o_full))


def main():
    args = common.parse_args("Hybrid certificate and per-cluster guarantee")
    ctx = common.load_ctx(args)
    from analysis.condition_block_corr import _range_bound_delta

    for layer in args.layers:
        groups = [group_data(q, k, v, qp, args.block_size, _range_bound_delta)
                  for q, k, v, qp in common.layer_groups(ctx, args, layer)]
        nf = len(FRACS)
        viol_t = [0] * nf
        viol_l = [0] * nf
        slack = [[] for _ in range(nf)]
        err_new = [[] for _ in range(nf)]
        err_old = [[] for _ in range(nf)]
        pc_viol = pc_total = 0

        for blocks, o_full, b_all in groups:
            nb = len(blocks)
            p_hat = torch.softmax(torch.tensor([b["z_hat"] for b in blocks]), dim=0)
            delta = torch.tensor([min(b["delta"], 80.0) for b in blocks])
            cosh_m1 = torch.cosh(delta) - 1.0
            tanh_h = torch.tanh(delta / 2.0)
            b_c = torch.tensor([b["b_c"] for b in blocks])
            score_new = p_hat * (2 * b_all * cosh_m1 + 2 * b_c * tanh_h)
            s_delta = float((p_hat * torch.cosh(delta)).sum())
            score_old = p_hat * (2 * b_all * cosh_m1 / s_delta + 2 * b_c * tanh_h)

            guar = p_hat * (b_all * torch.maximum(cosh_m1, torch.tensor((s_delta - 1.0) / s_delta))
                            + 2 * b_c * tanh_h)
            z_true_sum = sum(math.exp(b["logZ_true"]) for b in blocks)
            for i, blk in enumerate(blocks):
                pc_total += 1
                pc_viol += float(torch.norm(blk["num_true"] / z_true_sum
                                            - float(p_hat[i]) * blk["v_bar"])) > float(guar[i])

            for fi, frac in enumerate(FRACS):
                k = int(round(frac * nb))
                for scores, err_list in ((score_new, err_new), (score_old, err_old)):
                    order = torch.argsort(scores, descending=True)
                    sel = [False] * nb
                    for j in order[:k].tolist():
                        sel[j] = True
                    e = hybrid_error(blocks, sel, o_full)
                    err_list[fi].append(e)
                    if scores is score_new:
                        unsel = [i for i in range(nb) if not sel[i]]
                        T = float(sum(p_hat[i] * cosh_m1[i] for i in unsel))
                        ct = 2 * b_all * T / (1 + T) + float(
                            sum(2 * p_hat[i] * b_c[i] * tanh_h[i] for i in unsel))
                        viol_t[fi] += e > ct
                        viol_l[fi] += e > float(sum(score_new[i] for i in unsel))
                        slack[fi].append(e / max(ct, 1e-30))

        n = len(groups)
        print(f"\n===== layer {layer} ({n} groups) =====")
        print(f"literal per-cluster guarantee violations: {pc_viol}/{pc_total}")
        for fi, frac in enumerate(FRACS):
            sl = sorted(slack[fi])
            mn = torch.tensor(err_new[fi]).mean()
            mo = torch.tensor(err_old[fi]).mean()
            print(f"frac={frac:.2f}: cert violations tight={viol_t[fi]}/{n} loose={viol_l[fi]}/{n} | "
                  f"err/cert median={sl[len(sl) // 2]:.4f} max={sl[-1]:.4f} | "
                  f"err @ matched k: cert-score={mn:.4f} share={mo:.4f} (ratio {mn / max(float(mo), 1e-30):.3f})")


if __name__ == "__main__":
    main()
