import math
import time

import torch
from torch.nn import functional as F


def get_attention_map_after_rope(ctx, layer_idx, causal=True, dtype=None, device=None):
    """
    返回:
        scores: [nh, seq_len, seq_len] (softmax 前)
        attn:   [nh, seq_len, seq_len] (softmax 后)
    """
    if dtype is None:
        dtype = ctx.dtype
    if device is None:
        device = ctx.device

    Q = ctx.rope_qkv[layer_idx]["q"]  # [B, nh, seq, hd]
    K = ctx.rope_qkv[layer_idx]["k"]  # [B, nh, seq, hd]

    q = Q[0].to(dtype).to(device)  # [nh, seq, hd]
    k = K[0].to(dtype).to(device)  # [nh, seq, hd]

    hd = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)  # [nh, seq, seq]

    if causal:
        seq = scores.shape[1]
        mask = torch.triu(
            torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0)  # [1, seq, seq]
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return scores, attn


def optimize_alpha_star(
    ctx,
    layer_idx,
    head_idx,
    pos_list,
    training_steps,
    lr,
    mask,
    loss_type="logits_kl",
    device=None,
):
    """
    head_idx: int or list[int]
    pos_list: list[int]
    """
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    n_pos = len(pos_list)
    seq_len = mask.shape[-1]

    # [n_heads, n_pos, seq_len]
    a_param = torch.nn.Parameter(
        torch.randn(len(head_idx), n_pos, seq_len, device=device) * 0.1
    )
    a_param.retain_grad()

    opt = torch.optim.Adam([a_param], lr=lr)

    # constant part
    residual_attn_in = ctx.layer_input[layer_idx][0, pos_list].to(device)  # [n_pos, hidden_size]
    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)  # [nh, n_pos, hd]
    V_head = ctx.rope_qkv[layer_idx]["v"].to(device)[0][head_idx]  # [nh, seq, hd]
    layer = ctx.model.model.layers[layer_idx]

    if V_head.shape[1] != seq_len:
        raise ValueError(
            f"Mask seq_len ({seq_len}) does not match cached V seq_len ({V_head.shape[1]})."
        )

    gt_v = original[head_idx].detach().float()  # [n_heads, n_pos, hd]

    p_teacher = None
    logp_teacher = None
    p_teacher_v = None
    logp_teacher_v = None

    if loss_type == "logits_kl":
        with torch.no_grad():
            output = original.clone()  # [nh, n_pos, hd]
            hidden = layer.self_attn.o_proj(
                output.permute(1, 0, 2).reshape(len(pos_list), -1)
            )  # [n_pos, hidden_size]
            hidden = hidden + residual_attn_in
            gt_logits = ctx.model.lm_head(hidden)  # [n_pos, vocab_size]
            p_teacher = F.softmax(gt_logits.float(), dim=-1).detach()
            logp_teacher = F.log_softmax(gt_logits.float(), dim=-1).detach()
    elif loss_type == "v_kl":
        with torch.no_grad():
            p_teacher_v = F.softmax(gt_v, dim=-1).detach()
            logp_teacher_v = F.log_softmax(gt_v, dim=-1).detach()
    elif loss_type == "v_l2":
        pass
    else:
        raise ValueError(
            f"Unknown loss_type {loss_type}. Supported: logits_kl, v_l2, v_kl"
        )

    losses = []
    p_alpha = None

    for step in range(training_steps):
        alpha = F.softmax(a_param + mask, dim=-1)
        V_new = alpha @ V_head.float()  # [nh, n_pos, hd]
        V_new = V_new.to(original.dtype)

        if loss_type == "logits_kl":
            output = original.clone()
            output[head_idx] = V_new.to(V_head.dtype)

            hidden = output.permute(1, 0, 2).reshape(len(pos_list), -1)  # [n_pos, hidden_size]
            hidden = layer.self_attn.o_proj(hidden)
            hidden = hidden + residual_attn_in
            logits = ctx.model.lm_head(hidden)

            p_alpha = F.softmax(logits.float(), dim=-1)
            logp_student = F.log_softmax(logits.float(), dim=-1)
            loss = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1).mean()
        elif loss_type == "v_l2":
            loss = torch.norm(V_new.float() - gt_v, p=2, dim=-1).mean()
        else:  # loss_type == "v_kl"
            logp_student_v = F.log_softmax(V_new.float(), dim=-1)
            loss = (p_teacher_v * (logp_teacher_v - logp_student_v)).sum(dim=-1).mean()

        
        if step % 100 == 0:
            loss_v = loss.detach().float().cpu().item()
            losses.append((step, loss_v))
            print("step", step, "loss:", loss_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    alpha = F.softmax(a_param + mask, dim=-1)
    return alpha, p_alpha, p_teacher, losses


def build_kept_kv_cache(
    ctx,
    layer_idx,
    pos_list,
    head_idx,
    strategy,
    budget,
    seq_len,
    adaptive_budget,
    device=None,
):
    """
    Build compressed KV cache from retained token indices.

    Returns a dict containing:
      kept_indices: [nh, n_pos, max_keep] (Long, -1 for pad)
      keep_valid:   [nh, n_pos, max_keep] (Bool)
      kept_k:       [nh, n_pos, max_keep, hd]
      kept_v:       [nh, n_pos, max_keep, hd]
    """
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    mask = gen_mask(
        ctx=ctx,
        layer_idx=layer_idx,
        pos_list=pos_list,
        head_idx=head_idx,
        strategy=strategy,
        budget=budget,
        seq_len=seq_len,
        adaptive_budget=adaptive_budget,
    )

    visible = torch.isfinite(mask)
    nh, n_pos, _ = visible.shape
    keep_counts = visible.sum(dim=-1)
    max_keep = int(keep_counts.max().item())

    kept_indices = torch.full(
        (nh, n_pos, max_keep),
        -1,
        dtype=torch.long,
        device=device,
    )
    keep_valid = torch.zeros((nh, n_pos, max_keep), dtype=torch.bool, device=device)

    for h in range(nh):
        for i in range(n_pos):
            idx = torch.nonzero(visible[h, i], as_tuple=False).squeeze(-1)
            c = idx.numel()
            if c > 0:
                kept_indices[h, i, :c] = idx
                keep_valid[h, i, :c] = True

    k_full = ctx.rope_qkv[layer_idx]["k"].to(device)[0][head_idx].float()  # [nh, seq, hd]
    v_full = ctx.rope_qkv[layer_idx]["v"].to(device)[0][head_idx].float()  # [nh, seq, hd]

    gather_idx = kept_indices.clamp(min=0).unsqueeze(-1)  # [nh, n_pos, max_keep, 1]
    gather_idx = gather_idx.expand(-1, -1, -1, k_full.shape[-1])

    k_expand = k_full.unsqueeze(1).expand(-1, n_pos, -1, -1)
    v_expand = v_full.unsqueeze(1).expand(-1, n_pos, -1, -1)

    kept_k = torch.gather(k_expand, dim=2, index=gather_idx)
    kept_v = torch.gather(v_expand, dim=2, index=gather_idx)

    valid_4d = keep_valid.unsqueeze(-1)
    kept_k = torch.where(valid_4d, kept_k, torch.zeros_like(kept_k))
    kept_v = torch.where(valid_4d, kept_v, torch.zeros_like(kept_v))

    return {
        "kept_indices": kept_indices,
        "keep_valid": keep_valid,
        "kept_k": kept_k,
        "kept_v": kept_v,
    }


def optimize_alpha_star_on_kept_v(
    ctx,
    layer_idx,
    head_idx,
    pos_list,
    training_steps,
    lr,
    kept_v,
    keep_valid,
    loss_type="v_l2",
    device=None,
):
    """Optimize routing weights directly on compressed V cache."""
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    if loss_type != "v_l2":
        raise ValueError(
            f"optimize_alpha_star_on_kept_v only supports loss_type='v_l2', got {loss_type}."
        )

    n_heads = len(head_idx)
    n_pos = len(pos_list)

    if kept_v.shape[:3] != keep_valid.shape:
        raise ValueError(
            f"Shape mismatch: kept_v[:3]={kept_v.shape[:3]} vs keep_valid={keep_valid.shape}."
        )
    if kept_v.shape[0] != n_heads or kept_v.shape[1] != n_pos:
        raise ValueError(
            "kept_v shape is inconsistent with head_idx/pos_list. "
            f"Expected ({n_heads}, {n_pos}, *), got {kept_v.shape[:3]}"
        )

    kept_v = kept_v.to(device).float()
    keep_valid = keep_valid.to(device)

    a_param = torch.nn.Parameter(torch.randn_like(keep_valid, dtype=torch.float32) * 0.1)
    a_param.retain_grad()
    opt = torch.optim.Adam([a_param], lr=lr)

    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)
    gt_v = original[head_idx].detach().float()  # [nh, n_pos, hd]

    losses = []
    p_alpha = None
    p_teacher = None

    for step in range(training_steps):
        logits = a_param.masked_fill(~keep_valid, float("-inf"))
        alpha = F.softmax(logits, dim=-1)

        V_new = (alpha.unsqueeze(-1) * kept_v).sum(dim=2)
        loss = torch.norm(V_new - gt_v, p=2, dim=-1).mean()

        if step % 100 == 0:
            loss_v = loss.detach().float().cpu().item()
            losses.append((step, loss_v))
            print("step", step, "loss:", loss_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    logits = a_param.masked_fill(~keep_valid, float("-inf"))
    alpha = F.softmax(logits, dim=-1)
    return alpha, p_alpha, p_teacher, losses


def build_modified_attn_hidden_from_kept_v(
    ctx,
    layer_idx,
    head_idx,
    pos_list,
    alpha,
    kept_v,
    device=None,
):
    """Build patched attention hidden states using compressed V and optimized alpha."""
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    layer = ctx.model.model.layers[layer_idx]
    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)

    V_new = (alpha.to(device).unsqueeze(-1) * kept_v.to(device).float()).sum(dim=2)
    output = original.clone()
    output[head_idx] = V_new.to(output.dtype)

    attn_hidden = layer.self_attn.o_proj(output.permute(1, 0, 2).reshape(len(pos_list), -1))
    return attn_hidden


def build_qk_routing_alpha(ctx, layer_idx, head_idx, pos_list, mask, device=None):
    """Build baseline routing weights by applying original QK scores over the compressed KV mask."""
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    qk_scores, _ = get_attention_map_after_rope(
        ctx,
        layer_idx,
        causal=True,
        dtype=torch.float32,
        device=device,
    )
    qk_logits = qk_scores[head_idx][:, pos_list, :].to(torch.float32)
    if qk_logits.shape != mask.shape:
        raise ValueError(
            f"Baseline QK logits shape {qk_logits.shape} does not match mask shape {mask.shape}."
        )

    return F.softmax(qk_logits + mask.to(torch.float32), dim=-1)


def build_qk_routing_alpha_on_kept_kv(
    ctx,
    layer_idx,
    head_idx,
    pos_list,
    kept_k,
    keep_valid,
    device=None,
):
    """Build baseline routing weights on compressed kept-K cache."""
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    q_full = ctx.rope_qkv[layer_idx]["q"].to(device)[0][head_idx].float()  # [nh, seq, hd]
    q = q_full[:, pos_list, :]  # [nh, n_pos, hd]

    kept_k = kept_k.to(device).float()
    keep_valid = keep_valid.to(device)

    if kept_k.shape[:3] != keep_valid.shape:
        raise ValueError(
            f"Shape mismatch: kept_k[:3]={kept_k.shape[:3]} vs keep_valid={keep_valid.shape}."
        )
    if q.shape[0] != kept_k.shape[0] or q.shape[1] != kept_k.shape[1]:
        raise ValueError(
            "Q shape is inconsistent with kept_k. "
            f"Q={q.shape[:2]}, kept_k={kept_k.shape[:2]}"
        )

    hd = q.shape[-1]
    qk_logits = (q.unsqueeze(2) * kept_k).sum(dim=-1) / math.sqrt(hd)
    qk_logits = qk_logits.masked_fill(~keep_valid, float("-inf"))
    return F.softmax(qk_logits, dim=-1)


def gen_mask(
    ctx,
    layer_idx,
    pos_list,
    head_idx,
    strategy,
    budget,
    seq_len,
    adaptive_budget,
):
    """
    Return mask for alpha_param, with shape [nh, n_pos, seq_len]
    """
    device = ctx.device
    mask = torch.zeros(len(pos_list), seq_len, device=device)

    visible = int(seq_len * budget)
    if adaptive_budget:
        # do not compress frist 2 layers
        if layer_idx ==0 or layer_idx == 1:
            budget = 1.0
            visible = seq_len
    print(f"layer {layer_idx}: visible {visible} tokens for strategy {strategy} with budget {budget}")    
    if strategy == "recency":
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            if total_available > visible:
                mask[i, : total_available - visible] = float("-inf")
        mask = mask.unsqueeze(0).expand(len(head_idx), -1, -1)  # [nh, n_pos, seq_len]

    elif strategy == "random":
        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            if total_available > visible:
                idx = torch.randperm(total_available, device=device)[:visible]
                idx_list = idx.tolist()
                mask_list = list(set(range(total_available)) - set(idx_list))
                mask[i, mask_list] = float("-inf")
        mask = mask.unsqueeze(0).expand(len(head_idx), -1, -1)  # [nh, n_pos, seq_len]

    elif strategy == "attention_topk":
        attention_unnormalize, _ = get_attention_map_after_rope(
            ctx, layer_idx, causal=True, dtype=ctx.dtype, device=device
        )
        mask = torch.zeros(len(head_idx), len(pos_list), seq_len, device=device)
        for out_h, head in enumerate(head_idx):
            attention_score_head = attention_unnormalize[head]  # [seq, seq]
            for i, pos in enumerate(pos_list):
                total_available = pos + 1
                if total_available > visible:
                    topk = torch.topk(
                        attention_score_head[pos, :total_available], k=visible, largest=True
                    ).indices
                    idx_list = topk.tolist()
                    mask_list = list(set(range(total_available)) - set(idx_list))
                    mask[out_h, i, mask_list] = float("-inf")

    elif strategy == "h2o":
        recent_budget = visible // 2
        hh_budget = visible - recent_budget

        _, attention_score = get_attention_map_after_rope(
            ctx, layer_idx, causal=True, dtype=ctx.dtype, device=device
        )

        num_out_heads = len(head_idx)
        num_pos = len(pos_list)

        # final additive mask: 0 for visible, -inf for evicted/inaccessible
        mask = torch.zeros(num_out_heads, num_pos, seq_len, device=device)

        t1 = time.time()

        for out_h, head in enumerate(head_idx):
            attn = attention_score[head]  # [seq_len, seq_len]

            # accumulated attention score for each token j
            acc = torch.zeros(seq_len, device=device)

            # current membership in cache S_i
            in_cache = torch.zeros(seq_len, dtype=torch.bool, device=device)

            for i, pos in enumerate(pos_list):
                total_available = pos + 1

                acc[:total_available] += attn[pos, :total_available]

                in_cache[pos] = True

                cur_cache_size = int(in_cache[:total_available].sum().item())
                if cur_cache_size > visible:
                    cache_idx = torch.nonzero(in_cache[:total_available], as_tuple=False).squeeze(-1)
                    
                    recent_start = max(0, total_available - recent_budget)
                    hh_idx = cache_idx[cache_idx < recent_start]
                    assert len(hh_idx) > 0, f"No tokens in hh_idx for head {head} at position {pos} with recent_start {recent_start}"
                    victim_local = torch.argmin(acc[hh_idx])
                    victim = hh_idx[victim_local]

                    in_cache[victim] = False

                invisible_now = ~in_cache[:total_available]
                mask[out_h, i, :total_available][invisible_now] = float("-inf")

        t2 = time.time()
        print(f"[h2o_paper] mask build time: {t2 - t1:.4f}s")
    elif strategy == "sink":
        sink_keep = 4
        recent_keep = max(visible - sink_keep, 0)
        mask = torch.zeros(len(head_idx), len(pos_list), seq_len, device=device)

        for i, pos in enumerate(pos_list):
            total_available = pos + 1
            keep = torch.zeros(total_available, dtype=torch.bool, device=device)

            # Always keep the first sink_keep tokens (or all available if shorter).
            keep[: min(sink_keep, total_available)] = True

            # Keep the most recent tokens for the remaining budget.
            if recent_keep > 0:
                recent_start = max(0, total_available - recent_keep)
                keep[recent_start:total_available] = True

            mask[:, i, :total_available][:, ~keep] = float("-inf")
    elif strategy == "kvmerger":
        raise NotImplementedError("KVMerger strategy is not implemented yet.")

    else:
        raise ValueError(f"Unknown strategy {strategy}")

    for i, pos in enumerate(pos_list):
        mask[:, i, pos + 1 :] = float("-inf")

    return mask  # [n_heads, n_pos, seq_len]
