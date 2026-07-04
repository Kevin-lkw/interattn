"""PPL-vs-budget sweep with the Bennett condition (oracle sigma from block logits).

Wraps analysis.runner_cond_block, replacing only the condition computation in
_batched_hybrid_outputs_for_queries (cosh(delta) -> min{cosh, G(sigma, delta)}).
Everything else (selection rule shape, hybrid attention, PPL eval, budget
accounting) is identical, so curves are directly comparable to the saved
cosh-condition summaries under condition_block_runner/.

Run from repo root, e.g.:
  python llama/script/analysis/condition_bound/ppl_bennett.py \
      --model meta-llama/Llama-2-7b-hf --budget 0.05 --eps 0.01 0.03 0.1 0.3 1 3 \
      --output-dir llama/result/Llama-2-7b-hf/wikitext_0/condition_block_runner_bennett
All runner_cond_block flags are accepted.
"""

import math
import os
import sys

_SCRIPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPT_ROOT)

import torch

from analysis import runner_cond_block as rcb
from analysis.runner_cond_block import _gather_prefix


def _batched_hybrid_outputs_bennett(
    *,
    q_pos,
    pos_tensor,
    prefix,
    block_size,
    eps,
    delta_mode,
    share_selection_across_heads=False,
):
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = math.sqrt(head_dim)

    raw_prefix_len = pos_tensor[:, None] - prefix["block_starts"][None, :] + 1
    prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
    size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
    cluster_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)
    size_float = size.clamp_min(1).float()

    k_sum = _gather_prefix(prefix["k_cumsum"], prefix_idx)
    v_sum = _gather_prefix(prefix["v_cumsum"], prefix_idx)
    k_bar = k_sum / size_float.view(1, n_query, n_blocks, 1)
    v_bar = v_sum / size_float.view(1, n_query, n_blocks, 1)

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
        k_max = _gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = _gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_for_bounds = q_pos[:, :, None, :]
        upper_score = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(dim=-1) / scale
        lower_score = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(dim=-1) / scale
        delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = _gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)

    # --- Bennett first term: e^{b_C} <= min{cosh(delta), G(sigma, delta)} ---
    centered = (block_logits - s_c.unsqueeze(-1)).masked_fill(~token_visible.unsqueeze(0), 0.0)
    sigma2 = (centered ** 2).sum(dim=-1) / size_float.view(1, n_query, n_blocks)
    d64 = delta.double().clamp(max=80.0)
    s64 = sigma2.double()
    F = torch.ones_like(d64)
    ok = (d64 > 0) & (s64 > 0)
    F[ok] = (s64[ok] * torch.exp(d64[ok]) + d64[ok] ** 2 * torch.exp(-s64[ok] / d64[ok])) / (
        s64[ok] + d64[ok] ** 2)
    F = torch.minimum(F, torch.cosh(d64)).float()
    denom = (p_tensor * F).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (F - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta.clamp(max=80.0) / 2.0)
    )
    # --- end Bennett block (everything below is unchanged) ---

    if share_selection_across_heads:
        selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
        selected = selected.expand(n_heads, -1, -1)
    else:
        selected = (condition > eps) & cluster_exists.unsqueeze(0)

    token_selected = selected.unsqueeze(-1) & token_visible.unsqueeze(0)
    token_logits = block_logits.masked_fill(~token_selected, float("-inf"))
    cluster_logits = z_logits.masked_fill(selected | ~cluster_exists.unsqueeze(0), float("-inf"))

    token_max = token_logits.flatten(2).amax(dim=-1)
    cluster_max = cluster_logits.amax(dim=-1)
    max_logit = torch.maximum(token_max, cluster_max).clamp_min(-1e30)

    token_exp = torch.exp(token_logits - max_logit[:, :, None, None]).masked_fill(
        ~token_selected, 0.0)
    cluster_active = (~selected) & cluster_exists.unsqueeze(0)
    cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
        ~cluster_active, 0.0)
    normalizer = (token_exp.sum(dim=(2, 3)) + cluster_exp.sum(dim=2)).clamp_min(1e-30)

    token_num = torch.einsum("hqbt,hbtd->hqd", token_exp, prefix["v_block"])
    cluster_num = (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=2)
    output = (token_num + cluster_num) / normalizer.unsqueeze(-1)

    selected_tokens = (selected.long() * size.view(1, n_query, n_blocks)).sum()
    selected_clusters = selected.sum()
    clusters = cluster_exists.sum() * n_heads
    hybrid_tokens = selected_tokens + (cluster_active.sum())
    total_available = (pos_tensor.long() + 1).sum() * n_heads
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int(clusters.item()),
        "selected_clusters": int(selected_clusters.item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int(hybrid_tokens.item()),
        "total_available": int(total_available.item()),
    }
    return output, stats


if __name__ == "__main__":
    rcb._batched_hybrid_outputs_for_queries = _batched_hybrid_outputs_bennett
    rcb.main()
