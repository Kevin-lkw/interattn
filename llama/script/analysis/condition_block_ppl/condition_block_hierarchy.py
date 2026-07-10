"""
Shared hierarchical condition-block attention helpers.

The hierarchy is evaluated from coarse to fine block sizes, for example
128 -> 64 -> 32.  A block whose condition is <= eps is represented by its
average K/V.  A failed non-leaf block descends to children.  A failed leaf
block is expanded to token-level full attention.
"""

import math

import torch


def validate_block_sizes(block_sizes):
    sizes = [int(size) for size in block_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("block sizes must be positive")
    if sizes != sorted(sizes, reverse=True):
        raise ValueError("block sizes must be in descending order")
    for parent, child in zip(sizes, sizes[1:]):
        if parent % child != 0:
            raise ValueError("block hierarchy must be nested: parent must divide child")
    return sizes


def pad_blocks(x, block_size):
    n_heads, seq_len = x.shape[:2]
    tail_shape = x.shape[2:]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *tail_shape),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *tail_shape), n_blocks


def gather_prefix(prefix_tensor, prefix_idx):
    n_heads, n_blocks, block_size = prefix_tensor.shape[:3]
    tail_shape = prefix_tensor.shape[3:]
    n_query = prefix_idx.shape[0]
    expanded = prefix_tensor.unsqueeze(1).expand(
        n_heads, n_query, n_blocks, block_size, *tail_shape
    )
    gather_idx = prefix_idx.view(1, n_query, n_blocks, 1, *([1] * len(tail_shape))).expand(
        n_heads, n_query, n_blocks, 1, *tail_shape
    )
    return torch.gather(expanded, dim=3, index=gather_idx).squeeze(3)


def build_block_prefix_tensors(k_all, v_all, block_size):
    k_block, n_blocks = pad_blocks(k_all.float(), block_size)
    v_block, _ = pad_blocks(v_all.float(), block_size)
    device = k_all.device
    seq_len = k_all.shape[1]

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(
        n_blocks, block_size
    )
    valid_token = token_idx < seq_len

    valid_k = valid_token.view(1, n_blocks, block_size, 1)
    k_for_max = k_block.masked_fill(~valid_k, float("-inf"))
    k_for_min = k_block.masked_fill(~valid_k, float("inf"))
    v_norm = torch.norm(v_block, p=2, dim=-1)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    return {
        "block_size": int(block_size),
        "k_block": k_block,
        "v_block": v_block,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_cumsum": k_block.cumsum(dim=2),
        "v_cumsum": v_block.cumsum(dim=2),
        "k_prefix_max": k_for_max.cummax(dim=2).values,
        "k_prefix_min": k_for_min.cummin(dim=2).values,
        "v_norm_prefix_max": v_norm.cummax(dim=2).values,
        "block_starts": torch.arange(n_blocks, device=device) * block_size,
        "block_valid_counts": valid_token.sum(dim=1),
    }


def block_condition_parts(q_pos, pos_tensor, prefix, delta_mode):
    n_heads, n_query, head_dim = q_pos.shape
    block_size = int(prefix["block_size"])
    n_blocks = prefix["block_starts"].numel()
    scale = math.sqrt(head_dim)

    raw_prefix_len = pos_tensor[:, None] - prefix["block_starts"][None, :] + 1
    prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
    size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
    cluster_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)
    size_float = size.clamp_min(1).float()

    k_sum = gather_prefix(prefix["k_cumsum"], prefix_idx)
    v_sum = gather_prefix(prefix["v_cumsum"], prefix_idx)
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
        k_max = gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_for_bounds = q_pos[:, :, None, :]
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

    b_c = gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
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
        "size": size,
        "cluster_exists": cluster_exists,
        "block_logits": block_logits,
        "token_visible": token_visible,
    }


def hierarchical_batched_outputs_for_queries(
    *,
    q_pos,
    pos_tensor,
    prefixes,
    block_sizes,
    eps,
    delta_mode,
    share_selection_across_heads=True,
):
    block_sizes = validate_block_sizes(block_sizes)
    n_heads, n_query, head_dim = q_pos.shape
    max_parts = []
    cluster_parts = []
    token_parts = []
    active = None
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": 0,
        "total_available": int(((pos_tensor.long() + 1).sum() * n_heads).item()),
    }

    for level_idx, (block_size, prefix) in enumerate(zip(block_sizes, prefixes)):
        parts = block_condition_parts(q_pos, pos_tensor, prefix, delta_mode)
        condition = parts["condition"]
        cluster_exists = parts["cluster_exists"]
        if share_selection_across_heads:
            failed = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
            failed = failed.expand(n_heads, -1, -1)
        else:
            failed = (condition > eps) & cluster_exists.unsqueeze(0)

        if active is None:
            active = cluster_exists.unsqueeze(0).expand_as(failed)
        else:
            active = active[..., : cluster_exists.shape[-1]]
            active = active & cluster_exists.unsqueeze(0)

        is_leaf = level_idx == len(block_sizes) - 1
        stats["clusters"] += int(active.sum().item())

        accept_cluster = active & ~failed
        if accept_cluster.any():
            cluster_logits = parts["z_logits"].masked_fill(~accept_cluster, float("-inf"))
            max_parts.append(cluster_logits.amax(dim=-1))
            cluster_parts.append((cluster_logits, accept_cluster, parts["v_bar"]))
            stats["hybrid_tokens"] += int(accept_cluster.sum().item())

        failed_active = active & failed
        stats["selected_clusters"] += int(failed_active.sum().item())
        if is_leaf:
            token_active = failed_active.unsqueeze(-1) & parts["token_visible"].unsqueeze(0)
            token_logits = parts["block_logits"].masked_fill(~token_active, float("-inf"))
            max_parts.append(token_logits.flatten(2).amax(dim=-1))
            token_parts.append((token_logits, token_active, prefix["v_block"]))
            selected_tokens = int(
                (failed_active.long() * parts["size"].view(1, n_query, -1)).sum().item()
            )
            stats["selected_tokens"] += selected_tokens
            stats["hybrid_tokens"] += selected_tokens
        else:
            ratio = int(block_size // block_sizes[level_idx + 1])
            active = failed_active.repeat_interleave(ratio, dim=-1)

    max_logit = torch.stack(max_parts, dim=0).amax(dim=0)
    normalizer = torch.zeros_like(max_logit)
    numerator = torch.zeros(
        (*max_logit.shape, head_dim),
        device=q_pos.device,
        dtype=torch.float32,
    )
    for cluster_logits, cluster_active, v_bar in cluster_parts:
        cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
            ~cluster_active,
            0.0,
        )
        normalizer = normalizer + cluster_exp.sum(dim=-1)
        numerator = numerator + (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=2)
    for token_logits, token_active, v_block in token_parts:
        token_exp = torch.exp(token_logits - max_logit[:, :, None, None]).masked_fill(
            ~token_active,
            0.0,
        )
        normalizer = normalizer + token_exp.sum(dim=(2, 3))
        numerator = numerator + torch.einsum("hqbt,hbtd->hqd", token_exp, v_block)

    return numerator / normalizer.clamp_min(1e-30).unsqueeze(-1), stats
