"""Box vs diag_ell delta tightness on real model keys.

For sampled (head, query) pairs of real wikitext activations (stage-0 protocol:
`condition_bound.common.layer_groups`), computes per causal block the exact
delta, the box bound and the diag_ell bound, and reports per layer:

- median / p90 looseness ratio (bound / exact) for both bounds;
- the pairwise ratio diag_ell / box and the fraction of (query, block) pairs
  where diag_ell is strictly tighter;
- the per-query Spearman rank correlation of each bound with the exact delta
  across blocks (ordering quality, which is what selection actually consumes).

Defaults to the production model (Llama-3.1-8B) and block 32; pass
--model meta-llama/Llama-2-7b-hf to reproduce the stage-0 setting.
"""

import argparse
import math
import sys

import torch


def parse_args():
    local = argparse.ArgumentParser(description=__doc__)
    local.add_argument("--device", default="cuda:0")
    local.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    local.add_argument("--budget", type=float, default=0.03125, help="block_size = round(1/budget)")
    local.add_argument("--layers", type=int, nargs="+", default=[4, 10, 16, 22, 28])
    opts = local.parse_args()

    from ..condition_block_ppl import condition_block_corr as cbc

    sys.argv = [
        sys.argv[0],
        "--budget", str(opts.budget),
        "--layer", str(opts.layers[0]),
        "--device", opts.device,
        "--model", opts.model,
    ]
    args = cbc.parse_args()
    args.layers = opts.layers
    args.block_size = cbc._resolve_block_size(args)
    return args


def main():
    args = parse_args()
    from ..condition_block_ppl.condition_bound import common
    from ..condition_block_ppl.condition_block_corr import (
        _pearson,
        _range_bound_delta,
        _rankdata,
    )

    ctx = common.load_ctx(args)
    print(f"model={args.model} block_size={args.block_size}")
    overall = {"box": [], "diag_ell": [], "pair": [], "wins": 0, "n": 0}
    for layer in args.layers:
        ratios = {"box": [], "diag_ell": []}
        pair = []
        wins = 0
        n = 0
        sp = {"box": [], "diag_ell": []}
        groups = 0
        for q, k_head, _v_head, qpos in common.layer_groups(ctx, args, layer):
            groups += 1
            total = int(qpos) + 1
            scale = math.sqrt(q.numel())
            qf = q.float()
            exact_l, box_l, diag_l = [], [], []
            for start in range(0, total, args.block_size):
                end = min(start + args.block_size, total)
                kc = k_head[start:end].float()
                k_bar = kc.mean(dim=0)
                kd = kc - k_bar
                s_c = torch.dot(qf, k_bar) / scale
                exact_l.append(float(torch.mv(kd, qf).abs().max()) / scale)
                box_l.append(float(_range_bound_delta(qf, kc, s_c, scale)))
                w = kd.abs().max(dim=0).values.clamp_min(1e-6)
                rho = float((kd / w).norm(dim=-1).max())
                diag_l.append(rho * float((qf * w).norm()) / scale)
            exact = torch.tensor(exact_l)
            box = torch.tensor(box_l)
            diag = torch.tensor(diag_l)
            keep = exact > 0.01
            if int(keep.sum()) == 0:
                continue
            ratios["box"].extend((box[keep] / exact[keep]).tolist())
            ratios["diag_ell"].extend((diag[keep] / exact[keep]).tolist())
            pair.extend((diag[keep] / box[keep]).tolist())
            wins += int((diag[keep] < box[keep]).sum())
            n += int(keep.sum())
            if int(keep.sum()) >= 4:
                for name, t in (("box", box), ("diag_ell", diag)):
                    sp[name].append(
                        float(_pearson(_rankdata(t[keep]), _rankdata(exact[keep])))
                    )
        box_r = torch.tensor(ratios["box"])
        diag_r = torch.tensor(ratios["diag_ell"])
        pair_r = torch.tensor(pair)
        print(
            f"layer {layer:2d} ({groups} groups, {n} pairs): "
            f"box med={box_r.median():.2f} p90={box_r.quantile(0.9):.2f} | "
            f"diag_ell med={diag_r.median():.2f} p90={diag_r.quantile(0.9):.2f} | "
            f"diag/box med={pair_r.median():.3f}, diag tighter on {wins/n:.1%} | "
            f"spearman-vs-exact box={sum(sp['box'])/len(sp['box']):.3f} "
            f"diag={sum(sp['diag_ell'])/len(sp['diag_ell']):.3f}"
        )
        overall["box"].extend(ratios["box"])
        overall["diag_ell"].extend(ratios["diag_ell"])
        overall["pair"].extend(pair)
        overall["wins"] += wins
        overall["n"] += n
    box_r = torch.tensor(overall["box"])
    diag_r = torch.tensor(overall["diag_ell"])
    pair_r = torch.tensor(overall["pair"])
    print(
        f"ALL layers ({overall['n']} pairs): box med={box_r.median():.2f} | "
        f"diag_ell med={diag_r.median():.2f} | diag/box med={pair_r.median():.3f} | "
        f"diag tighter on {overall['wins']/overall['n']:.1%}"
    )


if __name__ == "__main__":
    main()
