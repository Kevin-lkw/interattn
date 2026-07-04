"""Estimator-correction ablation (negative result: kept for the record).

Mass estimators:  none  Z=|C|e^{s_C}   diag  +Vhat_diag/2   exact  +Var(s)/2
Value reps:       v_bar | v_lin = v_bar + Cov_i(v_i,s_i) | u_C (oracle)

Shows the mass/value errors of the average estimator anti-correlate: correcting
either side alone (or both at achievable order) does not reduce total error;
only the joint oracle does. See README.
"""

import math

import torch

import common

COMBOS = (
    ("none", "z0", "v_bar"), ("diag", "zd", "v_bar"), ("exact", "ze", "v_bar"),
    ("val", "z0", "v_lin"), ("joint", "ze", "v_lin"), ("joint-diag", "zd", "v_lin"),
    ("val-orc", "z0", "u_c"), ("joint-orc", "ze", "u_c"),
)


def group_eval(q, k_head, v_head, query_pos, block_size):
    total = int(query_pos) + 1
    scale = math.sqrt(q.numel())
    qf = q.float()
    vis_k = k_head[:total].float()
    vis_v = v_head[:total].float()
    o_full = (torch.softmax(torch.mv(vis_k, qf) / scale, dim=0).unsqueeze(-1) * vis_v).sum(dim=0).cpu()

    logz_true, zs = [], {"z0": [], "zd": [], "ze": []}
    vreps = {"v_bar": [], "v_lin": [], "u_c": []}
    resid_d, resid_e = [], []
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        mem = torch.arange(start, end, device=k_head.device)
        kc = k_head[mem].float()
        vc = v_head[mem].float()
        s = torch.mv(kc, qf) / scale
        s_c = float(s.mean())
        n = end - start
        var_exact = float(s.var(unbiased=False))
        var_diag = float(((qf ** 2) * kc.var(dim=0, unbiased=False)).sum() / (scale ** 2))
        b_c = float(torch.logsumexp(s - s_c, dim=0)) - math.log(n)
        logz_true.append(float(torch.logsumexp(s, dim=0)))
        zs["z0"].append(math.log(n) + s_c)
        zs["zd"].append(math.log(n) + s_c + var_diag / 2.0)
        zs["ze"].append(math.log(n) + s_c + var_exact / 2.0)
        resid_d.append(b_c - var_diag / 2.0)
        resid_e.append(b_c - var_exact / 2.0)
        v_bar = vc.mean(dim=0)
        vreps["v_bar"].append(v_bar)
        vreps["v_lin"].append(v_bar + (((s - s.mean()).unsqueeze(-1)) * (vc - v_bar)).mean(dim=0))
        vreps["u_c"].append((torch.softmax(s, dim=0).unsqueeze(-1) * vc).sum(dim=0))

    p_true = torch.softmax(torch.tensor(logz_true), dim=0)
    vreps = {k: torch.stack(v).cpu() for k, v in vreps.items()}
    out = {}
    for name, z_key, v_key in COMBOS:
        p_hat = torch.softmax(torch.tensor(zs[z_key]), dim=0)
        mass = float(((p_true - p_hat).abs() * torch.norm(vreps["u_c"], dim=-1)).sum())
        o_hat = (p_hat.unsqueeze(-1) * vreps[v_key]).sum(dim=0)
        out[name] = (mass, float(torch.norm(o_hat - o_full)))
    return out, resid_d, resid_e


def main():
    args = common.parse_args("Estimator correction ablation")
    ctx = common.load_ctx(args)
    for layer in args.layers:
        sums = {name: [0.0, 0.0] for name, _z, _v in COMBOS}
        rd, re = [], []
        for q, k, v, qp in common.layer_groups(ctx, args, layer):
            out, r_d, r_e = group_eval(q, k, v, qp, args.block_size)
            for name, (mass, tot) in out.items():
                sums[name][0] += mass
                sums[name][1] += tot
            rd.extend(r_d)
            re.extend(r_e)
        base_m, base_t = sums["none"]
        print(f"\n===== layer {layer} =====")
        for name, _z, _v in COMBOS:
            mass, tot = sums[name]
            print(f"{name:10s}: mass_err={mass:9.3f} ({mass / base_m:5.1%})  "
                  f"total ||o_hat-o||={tot:9.3f} ({tot / base_t:5.1%})")
        for name, r in (("diag", rd), ("exact", re)):
            t = torch.tensor(r)
            print(f"residual b_C - Vhat_{name}/2: median={t.median():+.4f} "
                  f"p90={t.quantile(0.9):+.4f} max={t.max():+.4f}")


if __name__ == "__main__":
    main()
