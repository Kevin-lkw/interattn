"""Decompose true cluster error into mass vs value channels; test b_C ~ Var(s)/2.

  mass_err_C  = |p_C - p_hat_C| * ||u_C||     value_err_C = p_hat_C * ||v_bar - u_C||
"""

import math

import torch

import common


def cluster_stats(q, k_head, v_head, query_pos, block_size):
    total = int(query_pos) + 1
    scale = math.sqrt(q.numel())
    qf = q.float()
    vis_k = k_head[:total].float()
    vis_v = v_head[:total].float()
    full_alpha = torch.softmax(torch.mv(vis_k, qf) / scale, dim=0)

    z_hat, z_true, rows = [], [], []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        mem = torch.arange(start, end, device=k_head.device)
        kc = k_head[mem].float()
        vc = v_head[mem].float()
        s = torch.mv(kc, qf) / scale
        s_c = float(s.mean())
        n = end - start
        b_c = float(torch.logsumexp(s - s_c, dim=0)) - math.log(n)
        var_half = float(s.var(unbiased=False)) / 2.0
        u_c = (torch.softmax(s, dim=0).unsqueeze(-1) * vc).sum(dim=0)
        v_bar = vc.mean(dim=0)
        o_c = (full_alpha[mem].unsqueeze(-1) * vc).sum(dim=0)
        z_hat.append(math.log(n) + s_c)
        z_true.append(float(torch.logsumexp(s, dim=0)))
        rows.append((u_c, v_bar, o_c, b_c, var_half))

    p_hat = torch.softmax(torch.tensor(z_hat), dim=0)
    p_true = torch.softmax(torch.tensor(z_true), dim=0)
    out = []
    for i, (u_c, v_bar, o_c, b_c, var_half) in enumerate(rows):
        out.append({
            "true_err": float(torch.norm(o_c - float(p_hat[i]) * v_bar)),
            "mass_err": float(abs(float(p_true[i] - p_hat[i])) * torch.norm(u_c)),
            "value_err": float(p_hat[i]) * float(torch.norm(v_bar - u_c)),
            "b_c": b_c,
            "var_half": var_half,
        })
    return out


def main():
    args = common.parse_args("Mass/value error decomposition")
    ctx = common.load_ctx(args)
    for layer in args.layers:
        rows = []
        for q, k, v, qp in common.layer_groups(ctx, args, layer):
            rows.extend(cluster_stats(q, k, v, qp, args.block_size))
        tm = sum(r["mass_err"] for r in rows)
        tv = sum(r["value_err"] for r in rows)
        b = torch.tensor([r["b_c"] for r in rows])
        vh = torch.tensor([r["var_half"] for r in rows])
        keep = b > 1e-6
        ratio = (vh[keep] / b[keep]).sort().values
        corr = torch.corrcoef(torch.stack([b[keep], vh[keep]]))[0, 1]
        n = ratio.numel()
        print(f"layer {layer}: sum true_err={sum(r['true_err'] for r in rows):.3f}  "
              f"mass={tm:.3f} ({100 * tm / (tm + tv):.0f}%)  value={tv:.3f} ({100 * tv / (tm + tv):.0f}%)")
        print(f"          b_C vs Var/2: corr={corr:.4f}  Var/2 over b_C: "
              f"p10={ratio[int(0.1 * n)]:.3f} median={ratio[n // 2]:.3f} p90={ratio[int(0.9 * n)]:.3f}")


if __name__ == "__main__":
    main()
