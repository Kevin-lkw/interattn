"""Condition-thresholded attention over K-similarity clusters.

This variant uses K similarity only to choose the cluster partition.  Each
cluster is still accepted or expanded by the same condition-style test used by
the contiguous block baselines.  There is no hierarchy or Triton path.
"""

import math

import torch

from .condition_ksim_cluster import resolve_num_clusters, spherical_kmeans_assign


def build_ksim_condition_prefix_tensors(
    k_head,
    v_head,
    *,
    cluster_size,
    kmeans_iters=4,
    max_clusters=None,
):
    """Build causal prefix statistics for one KV head's K-sim clusters."""

    if k_head.ndim != 2 or v_head.ndim != 2:
        raise ValueError("k_head and v_head must have shape [seq_len, head_dim]")
    if int(k_head.shape[0]) != int(v_head.shape[0]):
        raise ValueError("k_head and v_head must have the same seq_len")

    seq_len, head_dim = k_head.shape
    n_clusters = resolve_num_clusters(
        seq_len,
        cluster_size,
        max_clusters=max_clusters,
    )
    assignments = spherical_kmeans_assign(
        k_head,
        cluster_size=cluster_size,
        kmeans_iters=kmeans_iters,
        max_clusters=max_clusters,
    )
    device = k_head.device
    token_idx = torch.arange(seq_len, device=device, dtype=torch.long)

    count_updates = torch.zeros(seq_len, n_clusters, device=device, dtype=torch.float32)
    k_updates = torch.zeros(seq_len, n_clusters, head_dim, device=device, dtype=torch.float32)
    v_updates = torch.zeros(seq_len, n_clusters, head_dim, device=device, dtype=torch.float32)
    count_updates[token_idx, assignments] = 1.0
    k_updates[token_idx, assignments] = k_head.float()
    v_updates[token_idx, assignments] = v_head.float()

    valid = count_updates.bool()
    k_for_max = k_updates.masked_fill(~valid[..., None], float("-inf"))
    k_for_min = k_updates.masked_fill(~valid[..., None], float("inf"))
    v_norm = torch.norm(v_head.float(), p=2, dim=-1)
    v_norm_updates = torch.full(
        (seq_len, n_clusters),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    v_norm_updates[token_idx, assignments] = v_norm

    return {
        "cluster_size": int(cluster_size),
        "n_clusters": n_clusters,
        "assignments": assignments,
        "token_idx": token_idx,
        "k_tokens": k_head.float(),
        "v_tokens": v_head.float(),
        "counts_prefix": count_updates.cumsum(dim=0),
        "k_prefix": k_updates.cumsum(dim=0),
        "v_prefix": v_updates.cumsum(dim=0),
        "k_prefix_max": k_for_max.cummax(dim=0).values,
        "k_prefix_min": k_for_min.cummin(dim=0).values,
        "v_norm_prefix_max": v_norm_updates.cummax(dim=0).values,
    }


def ksim_condition_parts(q_pos, pos_tensor, prefix, delta_mode):
    """Return condition terms for arbitrary K-sim clusters."""

    n_heads, n_query, head_dim = q_pos.shape
    scale = math.sqrt(head_dim)
    counts = prefix["counts_prefix"][pos_tensor].float()
    cluster_exists = counts > 0
    counts_safe = counts.clamp_min(1.0)
    k_sum = prefix["k_prefix"][pos_tensor]
    v_sum = prefix["v_prefix"][pos_tensor]
    k_bar = k_sum / counts_safe[..., None]
    v_bar = v_sum / counts_safe[..., None]

    q_float = q_pos.float()
    s_c = torch.einsum("hqd,qcd->hqc", q_float, k_bar) / scale
    token_logits = None
    token_visible = prefix["token_idx"][None, :] <= pos_tensor[:, None]
    token_cluster = prefix["assignments"]

    if delta_mode == "exact":
        token_logits = torch.einsum("hqd,td->hqt", q_float, prefix["k_tokens"]) / scale
        centered = (token_logits - s_c[:, :, token_cluster]).abs()
        delta_vals = torch.full_like(s_c, float("-inf"))
        for cluster_idx in range(int(prefix["n_clusters"])):
            token_mask = (token_cluster == cluster_idx).view(1, 1, -1)
            active = token_visible.view(1, n_query, -1) & token_mask
            values = centered.masked_fill(~active, float("-inf")).amax(dim=-1)
            delta_vals[:, :, cluster_idx] = values
        delta = delta_vals.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    elif delta_mode == "range_bound":
        k_max = prefix["k_prefix_max"][pos_tensor]
        k_min = prefix["k_prefix_min"][pos_tensor]
        q_for_bounds = q_float[:, :, None, :]
        upper_score = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        lower_score = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = prefix["v_norm_prefix_max"][pos_tensor]
    b_c = b_c.masked_fill(~cluster_exists, 0.0).unsqueeze(0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(counts_safe).view(1, n_query, -1) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    return {
        "condition": condition,
        "z_logits": z_logits,
        "v_bar": v_bar,
        "counts": counts.long(),
        "cluster_exists": cluster_exists,
        "token_logits": token_logits,
        "token_visible": token_visible,
        "token_cluster": token_cluster,
    }


def condition_ksim_outputs_for_queries(
    *,
    q_pos,
    pos_tensor,
    prefix,
    eps,
    delta_mode,
    share_selection_across_heads=True,
):
    """Compute condition-thresholded K-sim cluster attention."""

    n_heads, n_query, head_dim = q_pos.shape
    parts = ksim_condition_parts(q_pos, pos_tensor, prefix, delta_mode)
    cluster_exists = parts["cluster_exists"]
    if share_selection_across_heads:
        failed = (parts["condition"].mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
        failed = failed.expand(n_heads, -1, -1)
    else:
        failed = (parts["condition"] > eps) & cluster_exists.unsqueeze(0)
    active = cluster_exists.unsqueeze(0).expand_as(failed)
    accept_cluster = active & ~failed
    failed_cluster = active & failed

    max_parts = []
    if accept_cluster.any():
        cluster_logits = parts["z_logits"].masked_fill(~accept_cluster, float("-inf"))
        max_parts.append(cluster_logits.amax(dim=-1))
    else:
        cluster_logits = None

    token_active = (
        failed_cluster[:, :, parts["token_cluster"]]
        & parts["token_visible"].unsqueeze(0)
    )
    if token_active.any():
        if parts["token_logits"] is None:
            token_logits_raw = (
                torch.einsum("hqd,td->hqt", q_pos.float(), prefix["k_tokens"])
                / math.sqrt(head_dim)
            )
        else:
            token_logits_raw = parts["token_logits"]
        token_logits = token_logits_raw.masked_fill(~token_active, float("-inf"))
        max_parts.append(token_logits.amax(dim=-1))
    else:
        token_logits = None

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    normalizer = torch.zeros_like(max_logit)
    numerator = torch.zeros(
        (*max_logit.shape, head_dim),
        device=q_pos.device,
        dtype=torch.float32,
    )
    if cluster_logits is not None:
        cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
            ~accept_cluster,
            0.0,
        )
        normalizer = normalizer + cluster_exp.sum(dim=-1)
        numerator = numerator + (cluster_exp.unsqueeze(-1) * parts["v_bar"]).sum(dim=2)
    if token_logits is not None:
        token_exp = torch.exp(token_logits - max_logit[:, :, None]).masked_fill(
            ~token_active,
            0.0,
        )
        normalizer = normalizer + token_exp.sum(dim=-1)
        numerator = numerator + torch.einsum("hqt,td->hqd", token_exp, prefix["v_tokens"])

    selected_tokens = int(
        (failed_cluster.long() * parts["counts"].view(1, n_query, -1)).sum().item()
    )
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int(active.sum().item()),
        "selected_clusters": int(failed_cluster.sum().item()),
        "selected_tokens": selected_tokens,
        "hybrid_tokens": int(accept_cluster.sum().item()) + selected_tokens,
        "total_available": int(((pos_tensor.long() + 1).sum() * n_heads).item()),
    }
    return numerator / normalizer.clamp_min(1e-30).unsqueeze(-1), stats
