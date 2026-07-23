"""Ball-bound eager selection for the generation-side condition-block runner.

`select_prompt_blocks_ball` mirrors
`condition_block_triton_impl.core._select_prompt_blocks_eager` with only the
delta computation replaced by the spherical bound

    delta_C(q) = ||q||_2 * r_C / sqrt(d_k),
    r_C = max over valid tokens of ||k_t - k_bar_C||_2.

The radius is computed once per layer from the cached prompt block summaries on
the first decode step and stored in the prefix dict (the box path likewise
reuses `k_max`/`k_min` built once per layer). Decode prompt blocks are fixed,
so the radius never changes within a sample.
"""

import torch


def ball_radius(prefix):
    """Exact per-(kv-head, block) radius, cached in the prefix dict."""
    radius = prefix.get("ball_radius")
    if radius is None:
        k_pages = prefix["k_block_attn"].float()  # (kv, n_blocks, block, d)
        k_bar = prefix["k_bar"].float()  # (kv, n_blocks, d)
        valid = prefix["valid_token"]  # (n_blocks, block)
        k_norm2 = k_pages.pow(2).sum(dim=-1)
        dot = torch.einsum("hbtd,hbd->hbt", k_pages, k_bar)
        dist2 = k_norm2 - 2.0 * dot + k_bar.pow(2).sum(dim=-1).unsqueeze(-1)
        dist2 = dist2.masked_fill(~valid.unsqueeze(0), float("-inf"))
        radius = dist2.amax(dim=-1).clamp_min(0.0).sqrt()  # (kv, n_blocks)
        prefix["ball_radius"] = radius
    return radius


def select_prompt_blocks_ball(q_grouped, prefix, eps):
    n_kv_heads, group_size, n_query, head_dim = q_grouped.shape
    scale = head_dim**0.5
    size = prefix["block_valid_counts"].view(1, 1, -1).long()
    cluster_exists = size > 0
    cluster_exists_view = cluster_exists.view(1, 1, 1, -1)
    size_float = size.clamp_min(1).float()

    k_bar = prefix["k_bar"].to(q_grouped.dtype)
    s_c = torch.einsum("gsqd,gbd->gsqb", q_grouped, k_bar) / scale

    # --- ball delta (the only change vs _select_prompt_blocks_eager) ---
    radius = ball_radius(prefix).to(q_grouped.dtype)
    q_norm = q_grouped.norm(dim=-1)  # (kv, group, n_query)
    delta = q_norm.unsqueeze(-1) * radius[:, None, None, :] / scale
    delta = delta.masked_fill(~cluster_exists_view, 0.0)
    # --- end ball delta ---

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
