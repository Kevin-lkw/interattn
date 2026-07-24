"""Ball / diag_ell eager selection for the generation-side condition-block runner.

Both variants mirror `condition_block_triton_impl.core._select_prompt_blocks_eager`
with only the delta computation replaced by a strict Cauchy-Schwarz bound:

- ball:      delta = ||q|| * r_C / sqrt(d),      r_C = max_t ||k_t - k_bar||
- diag_ell:  delta = rho_C * ||q * w_C|| / sqrt(d),
             w_C = max_t |k_t - k_bar| (elementwise),
             rho_C = max_t ||(k_t - k_bar) / w_C||

Per-block summaries are computed once per layer from the cached prompt block
pages on the first decode step and stored in the prefix dict (the box path
likewise reuses `k_max`/`k_min` built once per layer). Decode prompt blocks are
fixed, so the summaries never change within a sample.
"""

import os

import torch


def _masked_weighted_radius(prefix, inv_w2=None):
    """max_t ||(k_t - k_bar) * sqrt(weight)|| per (kv-head, block) in FP32.

    ``inv_w2=None`` uses unit weights (the plain ball radius). The distance uses
    ||k||^2 - 2 k.k_bar + ||k_bar||^2 to avoid materializing the centered pages.
    """
    k_pages = prefix["k_block_attn"].float()  # (kv, n_blocks, block, d)
    k_bar = prefix["k_bar"].float()  # (kv, n_blocks, d)
    valid = prefix["valid_token"]  # (n_blocks, block)
    if inv_w2 is None:
        k_norm2 = k_pages.pow(2).sum(dim=-1)
        dot = torch.einsum("hbtd,hbd->hbt", k_pages, k_bar)
        offset = k_bar.pow(2).sum(dim=-1)
    else:
        k_norm2 = torch.einsum("hbtd,hbd->hbt", k_pages.pow(2), inv_w2)
        dot = torch.einsum("hbtd,hbd->hbt", k_pages, k_bar * inv_w2)
        offset = (k_bar.pow(2) * inv_w2).sum(dim=-1)
    dist2 = k_norm2 - 2.0 * dot + offset.unsqueeze(-1)
    dist2 = dist2.masked_fill(~valid.unsqueeze(0), float("-inf"))
    return dist2.amax(dim=-1).clamp_min(0.0).sqrt()  # (kv, n_blocks)


def ball_radius(prefix):
    """Exact per-(kv-head, block) radius, cached in the prefix dict."""
    radius = prefix.get("ball_radius")
    if radius is None:
        radius = _masked_weighted_radius(prefix)
        prefix["ball_radius"] = radius
    return radius


def diag_ell_stats(prefix):
    """Per-(kv-head, block) `w` vector and `rho` scalar, cached in the prefix.

    ``CONDITION_BLOCK_BALL_W_DTYPE=bfloat16`` stores `w` in BF16 after a
    round-up bump. The bound stays strict for *any* positive weight vector as
    long as ``rho`` is computed with the same stored `w` (weighted
    Cauchy-Schwarz), so `rho` is derived from the cast value.
    """
    stats = prefix.get("diag_ell_stats")
    if stats is None:
        k_bar = prefix["k_bar"].float()
        w = torch.maximum(
            prefix["k_max"].float() - k_bar,
            k_bar - prefix["k_min"].float(),
        ).clamp_min(1e-6)  # (kv, n_blocks, d)
        if os.environ.get("CONDITION_BLOCK_BALL_W_DTYPE") == "bfloat16":
            w = (w * (1.0 + 2.0**-7)).to(torch.bfloat16)
        rho = _masked_weighted_radius(prefix, inv_w2=w.float().pow(-2))
        stats = (w, rho)
        prefix["diag_ell_stats"] = stats
    return stats


def _ball_delta(q_grouped, prefix, scale):
    radius = ball_radius(prefix).to(q_grouped.dtype)
    q_norm = q_grouped.norm(dim=-1)  # (kv, group, n_query)
    return q_norm.unsqueeze(-1) * radius[:, None, None, :] / scale


def _diag_ell_delta(q_grouped, prefix, scale):
    w, rho = diag_ell_stats(prefix)
    qw = torch.sqrt(
        torch.einsum("gsqd,gbd->gsqb", q_grouped.float().pow(2), w.float().pow(2)).clamp_min(0.0)
    ).to(q_grouped.dtype)
    return rho.to(q_grouped.dtype)[:, None, None, :] * qw / scale


def _select_prompt_blocks_with_delta(q_grouped, prefix, eps, delta_fn):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    size = prefix["block_valid_counts"].view(1, 1, -1).long()
    cluster_exists = size > 0
    cluster_exists_view = cluster_exists.view(1, 1, 1, -1)
    size_float = size.clamp_min(1).float()

    k_bar = prefix["k_bar"].to(q_grouped.dtype)
    s_c = torch.einsum("gsqd,gbd->gsqb", q_grouped, k_bar) / scale

    delta = delta_fn(q_grouped, prefix, scale)
    delta = delta.masked_fill(~cluster_exists_view, 0.0)

    b_c = prefix["v_norm_max"][:, None, None, :].expand(n_kv_heads, group_size, n_query, -1)
    b_c = b_c.masked_fill(~cluster_exists_view, 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).unsqueeze(2) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists_view, float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )

    selected = (condition.mean(dim=1, keepdim=True) > eps) & cluster_exists_view
    selected = selected.expand(n_kv_heads, group_size, n_query, -1)
    v_bar = prefix["v_bar"][:, None, None].expand(
        n_kv_heads,
        group_size,
        n_query,
        -1,
        head_dim,
    )
    return selected, z_logits, v_bar, size.view(1, -1), cluster_exists.view(1, -1)


def select_prompt_blocks_ball(q_grouped, prefix, eps):
    return _select_prompt_blocks_with_delta(q_grouped, prefix, eps, _ball_delta)


def select_prompt_blocks_diag_ell(q_grouped, prefix, eps):
    return _select_prompt_blocks_with_delta(q_grouped, prefix, eps, _diag_ell_delta)
