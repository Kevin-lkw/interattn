"""Stage 4: does the attention-weighted value centroid v* beat v_bar on real PPL?

For each window and eps we run the condition-block model twice at the SAME eps
(selection does not depend on the value representative, so the budget is identical
-> paired matched-budget comparison): once with the stock runner (v_bar) and once
with a monkeypatched hybrid that gives compressed clusters the attention-weighted
centroid  v* = sum_t softmax(q_bar . k_t) v_t,  q_bar = mean query direction of the
head over the window (one vector/head, leakage-free: weights use only visible tokens).

Only monkeypatches runner_cond_block._batched_hybrid_outputs_for_queries; no existing
file is modified.
"""

import argparse
import math
from types import SimpleNamespace

import torch

from .. import runner_cond_block as rcb
from ..multisample.common import (
    build_sample_context, load_model_and_tokens, prepare_sample, sample_starts,
)
from ...runner_utils import mean_nll_and_ppl


def _vstar_hybrid(*, q_pos, pos_tensor, prefix, block_size, eps, delta_mode,
                  share_selection_across_heads=False, force_first_last_blocks=False):
    """Copy of runner_cond_block._batched_hybrid_outputs_for_queries with v_bar ->
    attention-weighted centroid for the compressed clusters (only cluster_num changes)."""
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = math.sqrt(head_dim)

    raw_prefix_len = pos_tensor[:, None] - prefix["block_starts"][None, :] + 1
    prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
    size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
    cluster_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)
    size_float = size.clamp_min(1).float()

    k_sum = rcb._gather_prefix(prefix["k_cumsum"], prefix_idx)
    k_bar = k_sum / size_float.view(1, n_query, n_blocks, 1)

    s_c = (q_pos[:, :, None, :] * k_bar).sum(dim=-1) / scale
    block_logits = torch.einsum("hqd,hbtd->hqbt", q_pos, prefix["k_block"]) / scale
    token_visible = (
        prefix["valid_token"][None, :, :]
        & (prefix["token_idx"][None, :, :] <= pos_tensor[:, None, None])
    )

    if delta_mode == "exact":
        centered = (block_logits - s_c.unsqueeze(-1)).abs()
        delta = centered.masked_fill(~token_visible.unsqueeze(0), float("-inf")).amax(dim=-1)
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    elif delta_mode == "range_bound":
        k_max = rcb._gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = rcb._gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_for_bounds = q_pos[:, :, None, :]
        upper_score = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(dim=-1) / scale
        lower_score = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(dim=-1) / scale
        delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = rcb._gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    condition = rcb._condition_score_for_blocks(
        p_tensor=p_tensor, z_logits=z_logits, delta=delta, b_c=b_c, b_all=b_all,
        cluster_exists=cluster_exists,
    )
    if share_selection_across_heads:
        selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
        selected = selected.expand(n_heads, -1, -1)
    else:
        selected = (condition > eps) & cluster_exists.unsqueeze(0)

    if force_first_last_blocks:
        block_idx = torch.arange(n_blocks, device=pos_tensor.device)
        last_visible_block = cluster_exists.long().sum(dim=-1) - 1
        forced_by_query = cluster_exists & (
            (block_idx.unsqueeze(0) == 0)
            | (block_idx.unsqueeze(0) == last_visible_block.unsqueeze(1))
        )
        selected = selected | forced_by_query.unsqueeze(0).expand(n_heads, -1, -1)

    token_selected = selected.unsqueeze(-1) & token_visible.unsqueeze(0)
    token_logits = block_logits.masked_fill(~token_selected, float("-inf"))
    cluster_logits = z_logits.masked_fill(selected | ~cluster_exists.unsqueeze(0), float("-inf"))

    token_max = token_logits.flatten(2).amax(dim=-1)
    cluster_max = cluster_logits.amax(dim=-1)
    max_logit = torch.maximum(token_max, cluster_max).clamp_min(-1e30)

    token_exp = torch.exp(token_logits - max_logit[:, :, None, None]).masked_fill(~token_selected, 0.0)
    cluster_active = (~selected) & cluster_exists.unsqueeze(0)
    cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(~cluster_active, 0.0)
    normalizer = (token_exp.sum(dim=(2, 3)) + cluster_exp.sum(dim=2)).clamp_min(1e-30)

    # --- v* : attention-weighted centroid from the mean query direction (leakage-free) ---
    q_bar = q_pos.mean(dim=1)                                             # [h, d]
    star_logits = torch.einsum("hd,hbtd->hbt", q_bar, prefix["k_block"]) / scale
    star_logits = star_logits.unsqueeze(1).masked_fill(~token_visible.unsqueeze(0), float("-inf"))
    w_star = torch.nan_to_num(torch.softmax(star_logits, dim=-1), nan=0.0)   # [h,q,b,t]
    v_star = torch.einsum("hqbt,hbtd->hqbd", w_star, prefix["v_block"])      # [h,q,b,d]

    token_num = torch.einsum("hqbt,hbtd->hqd", token_exp, prefix["v_block"])
    cluster_num = (cluster_exp.unsqueeze(-1) * v_star).sum(dim=2)
    output = (token_num + cluster_num) / normalizer.unsqueeze(-1)

    selected_tokens = (selected.long() * size.view(1, n_query, n_blocks)).sum()
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int(cluster_exists.sum().item() * n_heads),
        "selected_clusters": int(selected.sum().item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int(selected_tokens.item() + cluster_active.sum().item()),
        "total_available": int(((pos_tensor.long() + 1).sum() * n_heads).item()),
    }
    return output, stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--ppl-protocol", default="aligned")
    p.add_argument("--block-size", type=int, default=10)
    p.add_argument("--delta-mode", default="exact", choices=["exact", "range_bound"])
    p.add_argument("--full-attention-layers", type=int, default=2)
    p.add_argument("--eps", type=float, nargs="+", default=[1.0, 0.3, 0.1])
    args = p.parse_args()
    if args.sample_stride is None:
        args.sample_stride = args.seq_len
    return args


def _nll(logits, labels):
    return float(mean_nll_and_ppl(logits, labels)[0])


def main():
    args = parse_args()
    model, tok, encoded, dtype, starts = load_model_and_tokens(args)
    runner_args = SimpleNamespace(seq_len=args.seq_len, block_size=args.block_size,
                                  delta_mode=args.delta_mode,
                                  full_attention_layers=args.full_attention_layers)
    layers = list(range(model.config.num_hidden_layers))
    orig = rcb._batched_hybrid_outputs_for_queries

    # per eps: lists of (nll_base, nll_star, budget, teacher_nll) over windows
    rec = {e: {"base": [], "star": [], "budget": [], "teacher": []} for e in args.eps}
    for si, start in enumerate(starts):
        ctx = build_sample_context(model, tok, encoded, dtype, args.device, start,
                                   args.seq_len, ppl_protocol=args.ppl_protocol)
        pos_list, model_inputs, labels, ref_logits, tnll, tppl = prepare_sample(ctx, args.seq_len)
        print(f"[window {si} start={start}] teacher ppl={tppl:.4f}")
        for e in args.eps:
            rcb._batched_hybrid_outputs_for_queries = orig
            lb, _, bud = rcb.run_for_eps(ctx, runner_args, e, layers, pos_list, model_inputs)
            nb = _nll(lb, labels)
            rcb._batched_hybrid_outputs_for_queries = _vstar_hybrid
            ls, _, _ = rcb.run_for_eps(ctx, runner_args, e, layers, pos_list, model_inputs)
            ns = _nll(ls, labels)
            budget = bud["aggregate"]["mean_budget_causal"]
            rec[e]["base"].append(nb); rec[e]["star"].append(ns)
            rec[e]["budget"].append(budget); rec[e]["teacher"].append(float(tnll))
            print(f"  eps={e:g} budget={budget:.4f}  base_ppl={math.exp(nb):.4f}  "
                  f"vstar_ppl={math.exp(ns):.4f}  dNLL={ns - nb:+.5f}")
            del lb, ls
    rcb._batched_hybrid_outputs_for_queries = orig

    print("\n===== paired matched-budget PPL (mean over windows) =====")
    print(f"{'eps':>6} {'budget':>8} {'teacher':>9} {'base':>9} {'vstar':>9} "
          f"{'dNLL':>9} {'ppl_ratio':>10}")
    for e in args.eps:
        r = rec[e]
        bnll = sum(r["base"]) / len(r["base"])
        snll = sum(r["star"]) / len(r["star"])
        bud = sum(r["budget"]) / len(r["budget"])
        tch = math.exp(sum(r["teacher"]) / len(r["teacher"]))
        print(f"{e:>6g} {bud:>8.4f} {tch:>9.4f} {math.exp(bnll):>9.4f} "
              f"{math.exp(snll):>9.4f} {snll - bnll:>+9.5f} {math.exp(snll) / math.exp(bnll):>10.4f}")


if __name__ == "__main__":
    main()
