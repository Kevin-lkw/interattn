"""Ball-bound delta for the teacher-forced condition-block PPL harness.

`_batched_hybrid_outputs_ball` is a copy of
`runner_cond_block._batched_hybrid_outputs_for_queries` with only the delta
computation replaced: instead of the coordinate-wise box bound
(`k_prefix_max`/`k_prefix_min`), it uses the spherical bound

    delta_C(q) = ||q||_2 * r_C / sqrt(d_k),
    r_C = max over visible tokens of ||k_t - k_bar_C||_2,

which is a valid upper bound on max_t |q.(k_t - k_bar_C)| / sqrt(d_k) by
Cauchy-Schwarz. The `delta_mode` argument is accepted for signature
compatibility and ignored. Everything else (condition score, selection rule,
hybrid attention, budget stats) is unchanged so curves are directly comparable
to the saved box-condition summaries.
"""

import math
import os

import torch

from ..condition_block_ppl.runner_cond_block import (
    _condition_score_for_blocks,
    _gather_prefix,
)


def ball_delta_for_queries(*, q_pos, k_bar, prefix, token_visible, cluster_exists, scale):
    """Per-(head, query, block) ball delta for causal block prefixes.

    ``k_bar`` is the prefix mean already gathered per query, so the radius is
    the max distance of the *visible* prefix tokens to that same mean. The
    distance uses ||k||^2 - 2 k.k_bar + ||k_bar||^2 to avoid materializing an
    (h, q, b, t, d) tensor.
    """
    k_block = prefix["k_block"]
    k_norm2 = k_block.pow(2).sum(dim=-1)  # (h, b, t)
    kbar_dot = torch.einsum("hqbd,hbtd->hqbt", k_bar, k_block)
    kbar_norm2 = k_bar.pow(2).sum(dim=-1)  # (h, q, b)
    dist2 = k_norm2[:, None] - 2.0 * kbar_dot + kbar_norm2.unsqueeze(-1)
    dist2 = dist2.masked_fill(~token_visible.unsqueeze(0), float("-inf"))
    radius = dist2.amax(dim=-1).clamp_min(0.0).sqrt()  # (h, q, b)
    delta = q_pos.norm(dim=-1)[:, :, None] * radius / scale
    return delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)


def diag_ell_delta_for_queries(
    *, q_pos, k_bar, prefix, prefix_idx, token_visible, cluster_exists, scale
):
    """Per-(head, query, block) diagonal-ellipsoid delta (strict bound).

    ``w_j = max_t |k_tj - k_bar_j|`` comes for free from the stored prefix
    cummax/cummin (``w = max(k_max - k_bar, k_bar - k_min)``), and
    ``rho = max_t ||(k_t - k_bar) / w||`` uses the same einsum decomposition as
    the ball radius, with 1/w^2 weights. Soundness is Cauchy-Schwarz in the
    w-weighted inner product: |q.d_t| <= ||q*w|| * ||d_t/w|| <= ||q*w|| * rho.
    """
    k_max = _gather_prefix(prefix["k_prefix_max"], prefix_idx)
    k_min = _gather_prefix(prefix["k_prefix_min"], prefix_idx)
    w = torch.maximum(k_max - k_bar, k_bar - k_min).clamp_min(1e-6)  # (h, q, b, d)
    inv_w2 = w.pow(-2)
    k_block = prefix["k_block"]
    # ||d_t / w||^2 = sum_j k_tj^2/w_j^2 - 2 sum_j k_tj k_bar_j/w_j^2 + sum_j k_bar_j^2/w_j^2
    t1 = torch.einsum("hbtd,hqbd->hqbt", k_block.pow(2), inv_w2)
    t2 = torch.einsum("hbtd,hqbd->hqbt", k_block, k_bar * inv_w2)
    t3 = (k_bar.pow(2) * inv_w2).sum(dim=-1)  # (h, q, b)
    norm2 = t1 - 2.0 * t2 + t3.unsqueeze(-1)
    norm2 = norm2.masked_fill(~token_visible.unsqueeze(0), float("-inf"))
    rho = norm2.amax(dim=-1).clamp_min(0.0).sqrt()  # (h, q, b)
    qw = (q_pos[:, :, None, :] * w).norm(dim=-1)  # (h, q, b)
    delta = rho * qw / scale
    return delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)


def _batched_hybrid_outputs_ball(
    *,
    q_pos,
    pos_tensor,
    prefix,
    block_size,
    eps,
    delta_mode,
    share_selection_across_heads=False,
    force_first_last_blocks=False,
    delta_variant="ball",
):
    del delta_mode  # always the ball/diag_ell variant below
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
    if os.environ.get("CONDITION_BLOCK_K_BAR_DTYPE") == "bfloat16":
        # Mirror the decode-side BF16 k_bar storage: the rounded center is used
        # for s_c AND for the deviation stats (w/rho), so delta stays a strict
        # bound around the stored center; only the center itself is approximate.
        k_bar = k_bar.to(torch.bfloat16).float()
    v_bar = v_sum / size_float.view(1, n_query, n_blocks, 1)

    s_c = (q_pos[:, :, None, :] * k_bar).sum(dim=-1) / scale
    block_logits = torch.einsum("hqd,hbtd->hqbt", q_pos, prefix["k_block"]) / scale
    token_visible = (
        prefix["valid_token"][None, :, :]
        & (prefix["token_idx"][None, :, :] <= pos_tensor[:, None, None])
    )

    # --- ball / diag_ell delta (the only change vs the box original) ---
    if delta_variant == "ball":
        delta = ball_delta_for_queries(
            q_pos=q_pos,
            k_bar=k_bar,
            prefix=prefix,
            token_visible=token_visible,
            cluster_exists=cluster_exists,
            scale=scale,
        )
    elif delta_variant == "diag_ell":
        delta = diag_ell_delta_for_queries(
            q_pos=q_pos,
            k_bar=k_bar,
            prefix=prefix,
            prefix_idx=prefix_idx,
            token_visible=token_visible,
            cluster_exists=cluster_exists,
            scale=scale,
        )
    else:
        raise ValueError(f"Unknown delta variant: {delta_variant}")
    # --- end ball / diag_ell delta ---

    b_c = _gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    condition = _condition_score_for_blocks(
        p_tensor=p_tensor,
        z_logits=z_logits,
        delta=delta,
        b_c=b_c,
        b_all=b_all,
        cluster_exists=cluster_exists,
    )
    if share_selection_across_heads:
        selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
        selected = selected.expand(n_heads, -1, -1)
    else:
        selected = (condition > eps) & cluster_exists.unsqueeze(0)

    forced = torch.zeros_like(selected)
    newly_forced = torch.zeros_like(selected)
    if force_first_last_blocks:
        block_idx = torch.arange(n_blocks, device=pos_tensor.device)
        last_visible_block = cluster_exists.long().sum(dim=-1) - 1
        forced_by_query = cluster_exists & (
            (block_idx.unsqueeze(0) == 0)
            | (block_idx.unsqueeze(0) == last_visible_block.unsqueeze(1))
        )
        forced = forced_by_query.unsqueeze(0).expand(n_heads, -1, -1)
        newly_forced = forced & ~selected
        selected = selected | forced

    token_selected = selected.unsqueeze(-1) & token_visible.unsqueeze(0)
    token_logits = block_logits.masked_fill(~token_selected, float("-inf"))
    cluster_logits = z_logits.masked_fill(selected | ~cluster_exists.unsqueeze(0), float("-inf"))

    token_max = token_logits.flatten(2).amax(dim=-1)
    cluster_max = cluster_logits.amax(dim=-1)
    max_logit = torch.maximum(token_max, cluster_max).clamp_min(-1e30)

    token_exp = torch.exp(token_logits - max_logit[:, :, None, None]).masked_fill(
        ~token_selected, 0.0
    )
    cluster_active = (~selected) & cluster_exists.unsqueeze(0)
    cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
        ~cluster_active, 0.0
    )
    normalizer = (
        token_exp.sum(dim=(2, 3)) + cluster_exp.sum(dim=2)
    ).clamp_min(1e-30)

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
    if force_first_last_blocks:
        stats.update(
            {
                "forced_clusters": int(forced.sum().item()),
                "forced_tokens": int(
                    (forced.long() * size.view(1, n_query, n_blocks)).sum().item()
                ),
                "newly_forced_clusters": int(newly_forced.sum().item()),
                "newly_forced_tokens": int(
                    (
                        newly_forced.long()
                        * size.view(1, n_query, n_blocks)
                    ).sum().item()
                ),
            }
        )
    return output, stats
