"""Shared post-Wo condition math and eager hybrid-attention implementation."""

import math

import torch

from ...sanity import grouped_query_heads


PRE_WO = "pre_wo"
POST_WO_SPECTRAL = "post_wo_spectral"
POST_WO_EXACT = "post_wo_exact"
CONDITION_VARIANTS = (PRE_WO, POST_WO_SPECTRAL, POST_WO_EXACT)

_SPECTRAL_NORM_CACHE = {}


def split_o_proj_weight(weight, num_heads, head_dim):
    """Return Wo column blocks as [num_heads, out_features, head_dim]."""
    if weight.ndim != 2:
        raise ValueError("o_proj weight must be a matrix")
    if int(weight.shape[1]) != int(num_heads * head_dim):
        raise ValueError(
            f"o_proj input width {weight.shape[1]} does not match "
            f"num_heads * head_dim = {num_heads * head_dim}"
        )
    return weight.detach().reshape(weight.shape[0], num_heads, head_dim).permute(1, 0, 2)


def head_spectral_norms(w_heads):
    """Compute exact spectral norms for a batch of Wo head blocks."""
    return torch.linalg.matrix_norm(w_heads.float(), ord=2, dim=(-2, -1))


def cached_layer_head_spectral_norms(weight, num_heads, head_dim):
    key = (id(weight), str(weight.device), int(num_heads), int(head_dim))
    cached = _SPECTRAL_NORM_CACHE.get(key)
    if cached is None:
        w_heads = split_o_proj_weight(weight, num_heads, head_dim)
        cached = head_spectral_norms(w_heads)
        _SPECTRAL_NORM_CACHE[key] = cached
    return cached


def projected_token_norms(v_heads, w_heads):
    """Compute ||W_h v_{h,t}|| without materializing hidden-size vectors."""
    vf = v_heads.float()
    wf = w_heads.float()
    gram = torch.einsum("hmd,hme->hde", wf, wf)
    norm_sq = torch.einsum("htd,hde,hte->ht", vf, gram, vf)
    return norm_sq.clamp_min(0.0).sqrt()


def value_norms_for_variant(v_heads, w_heads, variant, spectral_norms=None):
    if variant == PRE_WO:
        return torch.linalg.vector_norm(v_heads.float(), dim=-1)
    if variant == POST_WO_EXACT:
        if w_heads is None:
            raise ValueError("post_wo_exact requires Wo head blocks")
        return projected_token_norms(v_heads, w_heads)
    if variant == POST_WO_SPECTRAL:
        if spectral_norms is None:
            if w_heads is None:
                raise ValueError("post_wo_spectral requires Wo head blocks")
            spectral_norms = head_spectral_norms(w_heads)
        pre_norm = torch.linalg.vector_norm(v_heads.float(), dim=-1)
        return spectral_norms.float().unsqueeze(-1) * pre_norm
    raise ValueError(f"Unknown condition variant: {variant}")


def _pad_blocks(x, block_size, pad_value=0.0):
    n_heads, seq_len = x.shape[:2]
    tail_shape = x.shape[2:]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len:
        pad = torch.full(
            (n_heads, pad_len, *tail_shape),
            fill_value=pad_value,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *tail_shape), n_blocks


def _gather_prefix(prefix_tensor, prefix_idx):
    n_heads, n_blocks, block_size = prefix_tensor.shape[:3]
    tail_shape = prefix_tensor.shape[3:]
    n_query = prefix_idx.shape[0]
    expanded = prefix_tensor.unsqueeze(1).expand(
        n_heads, n_query, n_blocks, block_size, *tail_shape
    )
    gather_idx = prefix_idx.view(
        1, n_query, n_blocks, 1, *([1] * len(tail_shape))
    ).expand(n_heads, n_query, n_blocks, 1, *tail_shape)
    return torch.gather(expanded, dim=3, index=gather_idx).squeeze(3)


def build_block_prefix(k_heads, v_heads, value_norms, block_size):
    """Build causal-prefix summaries for one GQA group."""
    k_block, n_blocks = _pad_blocks(k_heads.float(), block_size)
    v_block, _ = _pad_blocks(v_heads.float(), block_size)
    norm_block, _ = _pad_blocks(value_norms.float(), block_size, pad_value=float("-inf"))
    device = k_heads.device
    seq_len = int(k_heads.shape[1])

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(
        n_blocks, block_size
    )
    valid_token = token_idx < seq_len
    valid_k = valid_token.view(1, n_blocks, block_size, 1)
    k_for_max = k_block.masked_fill(~valid_k, float("-inf"))
    k_for_min = k_block.masked_fill(~valid_k, float("inf"))
    norm_block = norm_block.masked_fill(
        ~valid_token.view(1, n_blocks, block_size), float("-inf")
    )

    return {
        "k_block": k_block,
        "v_block": v_block,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_cumsum": k_block.cumsum(dim=2),
        "v_cumsum": v_block.cumsum(dim=2),
        "k_prefix_max": k_for_max.cummax(dim=2).values,
        "k_prefix_min": k_for_min.cummin(dim=2).values,
        "value_norm_prefix_max": norm_block.cummax(dim=2).values,
        "block_starts": torch.arange(n_blocks, device=device) * block_size,
        "block_valid_counts": valid_token.sum(dim=1),
    }


def compute_condition_data(q_pos, pos_tensor, prefix, block_size, delta_mode):
    """Compute block condition and retain tensors used by offline diagnostics."""
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = int(prefix["block_starts"].numel())
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
        delta = centered.masked_fill(
            ~token_visible.unsqueeze(0), float("-inf")
        ).amax(dim=-1)
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    elif delta_mode == "range_bound":
        k_max = _gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = _gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_for_bounds = q_pos[:, :, None, :]
        upper = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        lower = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        delta = torch.maximum((upper - s_c).abs(), (lower - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = _gather_prefix(
        prefix["value_norm_prefix_max"].unsqueeze(-1), prefix_idx
    ).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_hat = torch.softmax(z_logits, dim=-1)
    f_value = torch.cosh(delta)
    s_f = (p_hat * f_value).sum(dim=-1).clamp_min(1e-30)
    condition = p_hat * (
        2.0 * b_all.unsqueeze(-1) * (f_value - 1.0) / s_f.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )

    return {
        "size": size,
        "cluster_exists": cluster_exists,
        "v_bar": v_bar,
        "s_c": s_c,
        "block_logits": block_logits,
        "token_visible": token_visible,
        "delta": delta,
        "b_c": b_c,
        "b_all": b_all,
        "z_logits": z_logits,
        "p_hat": p_hat,
        "f_value": f_value,
        "s_f": s_f,
        "condition": condition,
    }


def hybrid_output_from_condition(data, prefix, eps, share_selection_across_heads=True):
    condition = data["condition"]
    cluster_exists = data["cluster_exists"]
    if share_selection_across_heads:
        selected = (
            condition.mean(dim=0, keepdim=True) > float(eps)
        ) & cluster_exists.unsqueeze(0)
        selected = selected.expand(condition.shape[0], -1, -1)
    else:
        selected = (condition > float(eps)) & cluster_exists.unsqueeze(0)

    token_selected = selected.unsqueeze(-1) & data["token_visible"].unsqueeze(0)
    token_logits = data["block_logits"].masked_fill(~token_selected, float("-inf"))
    cluster_logits = data["z_logits"].masked_fill(
        selected | ~cluster_exists.unsqueeze(0), float("-inf")
    )

    token_max = token_logits.flatten(2).amax(dim=-1)
    cluster_max = cluster_logits.amax(dim=-1)
    max_logit = torch.maximum(token_max, cluster_max).clamp_min(-1e30)
    token_exp = torch.exp(
        token_logits - max_logit[:, :, None, None]
    ).masked_fill(~token_selected, 0.0)
    cluster_active = (~selected) & cluster_exists.unsqueeze(0)
    cluster_exp = torch.exp(
        cluster_logits - max_logit[:, :, None]
    ).masked_fill(~cluster_active, 0.0)
    normalizer = (
        token_exp.sum(dim=(2, 3)) + cluster_exp.sum(dim=2)
    ).clamp_min(1e-30)

    token_num = torch.einsum("hqbt,hbtd->hqd", token_exp, prefix["v_block"])
    cluster_num = (cluster_exp.unsqueeze(-1) * data["v_bar"]).sum(dim=2)
    output = (token_num + cluster_num) / normalizer.unsqueeze(-1)

    selected_tokens = (selected.long() * data["size"].unsqueeze(0)).sum()
    selected_clusters = selected.sum()
    clusters = cluster_exists.sum() * int(condition.shape[0])
    hybrid_tokens = selected_tokens + cluster_active.sum()
    total_available = (data["size"].sum(dim=-1).sum() * int(condition.shape[0]))
    stats = {
        "rows": int(condition.shape[0] * condition.shape[1]),
        "clusters": int(clusters.item()),
        "selected_clusters": int(selected_clusters.item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int(hybrid_tokens.item()),
        "total_available": int(total_available.item()),
    }
    return output, selected, stats


def build_condition_block_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    block_size,
    eps,
    delta_mode,
    variant,
):
    """Build one layer patch using a pre- or post-Wo condition variant."""
    if variant not in CONDITION_VARIANTS:
        raise ValueError(f"Unknown condition variant: {variant}")
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    output_dtype = artifacts["attn_output"].dtype
    n_heads = int(q_all.shape[0])
    head_dim = int(q_all.shape[-1])
    pos_tensor = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    output = torch.empty(
        n_heads,
        len(pos_list),
        head_dim,
        device=ctx.device,
        dtype=torch.float32,
    )

    layer = ctx.model.model.layers[layer_idx]
    w_all = split_o_proj_weight(layer.self_attn.o_proj.weight, n_heads, head_dim)
    spectral_all = None
    if variant == POST_WO_SPECTRAL:
        spectral_all = cached_layer_head_spectral_norms(
            layer.self_attn.o_proj.weight, n_heads, head_dim
        )

    merged_stats = {}
    groups = grouped_query_heads(
        list(range(n_heads)), ctx.model_config, num_kv_heads=k_all.shape[0]
    )
    for kv_head, _out_indices, query_heads in groups:
        q_pos = q_all[query_heads][:, pos_tensor, :].float()
        k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        w_group = w_all[query_heads]
        spectral_group = None if spectral_all is None else spectral_all[query_heads]
        value_norms = value_norms_for_variant(
            v_group, w_group, variant, spectral_norms=spectral_group
        )
        prefix = build_block_prefix(k_group, v_group, value_norms, block_size)
        data = compute_condition_data(
            q_pos, pos_tensor, prefix, block_size, delta_mode
        )
        group_output, _selected, group_stats = hybrid_output_from_condition(
            data, prefix, eps, share_selection_across_heads=True
        )
        output[query_heads] = group_output
        for key, value in group_stats.items():
            merged_stats[key] = int(merged_stats.get(key, 0)) + int(value)

    proj_dtype = layer.self_attn.o_proj.weight.dtype
    with torch.no_grad():
        patch_hidden = layer.self_attn.o_proj(
            output.to(output_dtype)
            .permute(1, 0, 2)
            .reshape(len(pos_list), -1)
            .to(ctx.device, dtype=proj_dtype)
        )
    return patch_hidden.detach(), merged_stats


def exact_block_contribution_errors(data, prefix):
    """Return pre-Wo block contribution errors [head, query, block, dim]."""
    visible = data["token_visible"].unsqueeze(0)
    logits = data["block_logits"].masked_fill(~visible, float("-inf"))
    flat_alpha = torch.softmax(logits.flatten(2), dim=-1)
    alpha = flat_alpha.reshape_as(logits).masked_fill(~visible, 0.0)
    exact_contrib = torch.einsum("hqbt,hbtd->hqbd", alpha, prefix["v_block"])
    approx_contrib = data["p_hat"].unsqueeze(-1) * data["v_bar"]
    return approx_contrib - exact_contrib
