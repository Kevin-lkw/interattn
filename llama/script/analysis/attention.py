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
    loss_type="v_l2",
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

    with torch.no_grad():
        alpha = F.softmax(a_param + mask, dim=-1).detach()
        if p_alpha is not None:
            p_alpha = p_alpha.detach()

    return alpha, p_alpha, p_teacher, losses


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


def gen_mask_h2o_with_belong(
    ctx,
    layer_idx,
    pos_list,
    head_idx,
    budget,
    seq_len,
    adaptive_budget,
    merge_metric="k",
    return_hh_sumv=False,
    return_hh_sumk=False,
):
    """
    H2O-only mask generator with belong mapping.

    Returns:
        mask:   [n_heads, n_pos, seq_len], additive mask (0 or -inf)
        belong: [n_heads, n_pos, seq_len], belong indices for keys <= current pos,
                and -1 for keys > current pos.
        count_mtx: [n_heads, n_pos, seq_len], merged-set sizes on parent indices.
        (optional) hh_sumv_idx, hh_sumv_val:
            hh_sumv_idx[h][i] -> LongTensor [k_i] visible heavy-hitter key indices at pos i.
            hh_sumv_val[h][i] -> FloatTensor [k_i, hd] summed V for those heavy hitters.
        (optional) hh_sumk_val:
            hh_sumk_val[h][i] -> FloatTensor [k_i, hd] summed K for same hh_sumv_idx[h][i].

    Rule:
    - If token x is evicted, assign belong[x] = y where y is the kept heavy-hitter
      token (older-than-recent window) with maximum dot-product similarity under
      the selected merge_metric ("k" or "v").
    - If token x is kept (or no valid heavy hitter exists), belong[x] = x.
    - Transitive merge is resolved by DSU root when emitting belong rows.
    """
    device = ctx.device
    if isinstance(head_idx, int):
        head_idx = [head_idx]
    if merge_metric not in {"k", "v"}:
        raise ValueError(f"Unsupported merge_metric {merge_metric}. Expected 'k' or 'v'.")

    visible = int(seq_len * budget)
    if adaptive_budget and (layer_idx == 0 or layer_idx == 1):
        budget = 1.0
        visible = seq_len

    recent_budget = visible // 2

    print(
        f"layer {layer_idx}: visible {visible} tokens for strategy h2o_with_belong with budget {budget}"
    )

    k_all = ctx.rope_qkv[layer_idx]["k"].to(device=device, dtype=torch.float32)[0][head_idx]
    v_all = ctx.rope_qkv[layer_idx]["v"].to(device=device, dtype=torch.float32)[0][head_idx]
    _, attention_score = get_attention_map_after_rope(
        ctx, layer_idx, causal=True, dtype=ctx.dtype, device=device
    )

    num_out_heads = len(head_idx)
    num_pos = len(pos_list)

    mask = torch.zeros(num_out_heads, num_pos, seq_len, device=device)
    belong = torch.full(
        (num_out_heads, num_pos, seq_len),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    count_mtx = torch.zeros(num_out_heads, num_pos, seq_len, device=device, dtype=torch.long)
    hh_sumv_idx = None
    hh_sumv_val = None
    hh_sumk_val = None
    if return_hh_sumv:
        hh_sumv_idx = [[None for _ in range(num_pos)] for _ in range(num_out_heads)]
        hh_sumv_val = [[None for _ in range(num_pos)] for _ in range(num_out_heads)]
    if return_hh_sumk:
        if hh_sumv_idx is None:
            hh_sumv_idx = [[None for _ in range(num_pos)] for _ in range(num_out_heads)]
        hh_sumk_val = [[None for _ in range(num_pos)] for _ in range(num_out_heads)]
    t1 = time.time()

    for out_h, head in enumerate(head_idx):
        attn = attention_score[head]
        k_head = k_all[out_h]
        v_head = v_all[out_h]

        acc = torch.zeros(seq_len, device=device)
        in_cache = torch.zeros(seq_len, dtype=torch.bool, device=device)
        parent = torch.arange(seq_len, device=device, dtype=torch.long)
        count = torch.ones(seq_len, dtype=torch.long, device=device)
        group_sum_v = torch.zeros_like(v_head)
        group_sum_k = torch.zeros_like(k_head)

        for i, pos in enumerate(pos_list):
            total_available = pos + 1

            acc[:total_available] += attn[pos, :total_available]
            in_cache[pos] = True
            group_sum_v[pos] = v_head[pos]
            group_sum_k[pos] = k_head[pos]

            cur_cache_size = int(in_cache[:total_available].sum().item())
            if cur_cache_size > visible:
                cache_idx = torch.nonzero(
                    in_cache[:total_available], as_tuple=False
                ).squeeze(-1)

                recent_start = max(0, total_available - recent_budget)
                hh_idx = cache_idx[cache_idx < recent_start]
                if len(hh_idx) == 0:
                    victim = int(cache_idx[0].item())
                else:
                    victim_local = torch.argmin(acc[hh_idx])
                    victim = int(hh_idx[victim_local].item())

                in_cache[victim] = False

                cache_idx_after = torch.nonzero(
                    in_cache[:total_available], as_tuple=False
                ).squeeze(-1)
                hh_kept_idx = cache_idx_after[cache_idx_after < recent_start]

                if len(hh_kept_idx) > 0:
                    if merge_metric == "k":
                        sims = torch.matmul(k_head[hh_kept_idx], k_head[victim])
                    else:
                        sims = torch.matmul(v_head[hh_kept_idx], v_head[victim])
                    y = int(hh_kept_idx[torch.argmax(sims)].item())
                else:
                    assert False, f"No heavy hitter to assign for victim {victim} at position {pos}"
                

                parent[victim] = y
                count[y] += count[victim]
                count[victim] = 0
                group_sum_v[y] += group_sum_v[victim]
                group_sum_v[victim].zero_()
                group_sum_k[y] += group_sum_k[victim]
                group_sum_k[victim].zero_()

            invisible_now = ~in_cache[:total_available]
            mask[out_h, i, :total_available][invisible_now] = float("-inf")

            recent_start = max(0, total_available - recent_budget)
            visible_now = in_cache[:total_available]
            hh_visible = visible_now.clone()
            hh_visible[recent_start:total_available] = False
            hh_idx = torch.nonzero(hh_visible, as_tuple=False).squeeze(-1)

            # roots = torch.arange(total_available, device=device, dtype=torch.long)
            # for x in range(total_available):
            #     roots[x] = find_root(int(roots[x].item()))
            belong[out_h, i, :total_available] = parent[:total_available]
            count_mtx[out_h, i, :total_available] = count[:total_available]
            if return_hh_sumv:
                hh_sumv_idx[out_h][i] = hh_idx.clone()
                hh_sumv_val[out_h][i] = group_sum_v[hh_idx].clone()
            elif return_hh_sumk:
                hh_sumv_idx[out_h][i] = hh_idx.clone()
            if return_hh_sumk:
                hh_sumk_val[out_h][i] = group_sum_k[hh_idx].clone()

    t2 = time.time()
    print(f"[h2o_with_belong] mask+belong build time: {t2 - t1:.4f}s")

    for i, pos in enumerate(pos_list):
        mask[:, i, pos + 1 :] = float("-inf")
        belong[:, i, pos + 1 :] = -1

    if return_hh_sumv and return_hh_sumk:
        return mask, belong, count_mtx, hh_sumv_idx, hh_sumv_val, hh_sumk_val
    if return_hh_sumv:
        return mask, belong, count_mtx, hh_sumv_idx, hh_sumv_val
    if return_hh_sumk:
        return mask, belong, count_mtx, hh_sumv_idx, hh_sumk_val
    return mask, belong, count_mtx


def gen_mask_h2o_with_belong_all(
    ctx,
    layer_idx,
    pos_list,
    head_idx,
    budget,
    seq_len,
    adaptive_budget,
    merge_metric="k",
):
    """
    H2O mask generator with belong mapping for all visible tokens.

        Difference from gen_mask_h2o_with_belong:
    - Eviction policy remains H2O-style (evict from heavy-hitter part, not recent window).
    - Routing refinement is applied to all visible tokens via count_mtx.
        - When assigning an evicted token x, merge target y is selected from all
            currently kept tokens by maximum similarity under merge_metric ("k" or "v").

    Returns:
        mask:   [n_heads, n_pos, seq_len], additive mask (0 or -inf)
        belong: [n_heads, n_pos, seq_len], parent index for keys <= current pos,
                and -1 for keys > current pos.
        count_mtx: [n_heads, n_pos, seq_len], merged-set sizes on parent indices.
    """
    device = ctx.device
    if isinstance(head_idx, int):
        head_idx = [head_idx]
    if merge_metric not in {"k", "v"}:
        raise ValueError(f"Unsupported merge_metric {merge_metric}. Expected 'k' or 'v'.")

    visible = int(seq_len * budget)
    if adaptive_budget and (layer_idx == 0 or layer_idx == 1):
        budget = 1.0
        visible = seq_len

    recent_budget = visible // 2

    print(
        f"layer {layer_idx}: visible {visible} tokens for strategy h2o_with_belong_all with budget {budget}"
    )

    k_all = ctx.rope_qkv[layer_idx]["k"].to(device=device, dtype=torch.float32)[0][head_idx]
    v_all = ctx.rope_qkv[layer_idx]["v"].to(device=device, dtype=torch.float32)[0][head_idx]
    _, attention_score = get_attention_map_after_rope(
        ctx, layer_idx, causal=True, dtype=ctx.dtype, device=device
    )

    num_out_heads = len(head_idx)
    num_pos = len(pos_list)

    mask = torch.zeros(num_out_heads, num_pos, seq_len, device=device)
    belong = torch.full(
        (num_out_heads, num_pos, seq_len),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    count_mtx = torch.zeros(num_out_heads, num_pos, seq_len, device=device, dtype=torch.long)

    t1 = time.time()

    for out_h, head in enumerate(head_idx):
        attn = attention_score[head]
        k_head = k_all[out_h]
        v_head = v_all[out_h]

        acc = torch.zeros(seq_len, device=device)
        in_cache = torch.zeros(seq_len, dtype=torch.bool, device=device)
        parent = torch.arange(seq_len, device=device, dtype=torch.long)
        count = torch.ones(seq_len, dtype=torch.long, device=device)

        for i, pos in enumerate(pos_list):
            total_available = pos + 1

            acc[:total_available] += attn[pos, :total_available]
            in_cache[pos] = True

            cur_cache_size = int(in_cache[:total_available].sum().item())
            if cur_cache_size > visible:
                cache_idx = torch.nonzero(
                    in_cache[:total_available], as_tuple=False
                ).squeeze(-1)

                recent_start = max(0, total_available - recent_budget)
                hh_idx = cache_idx[cache_idx < recent_start]
                if len(hh_idx) == 0:
                    victim = int(cache_idx[0].item())
                else:
                    victim_local = torch.argmin(acc[hh_idx])
                    victim = int(hh_idx[victim_local].item())

                in_cache[victim] = False

                kept_idx = torch.nonzero(
                    in_cache[:total_available], as_tuple=False
                ).squeeze(-1)
                if len(kept_idx) == 0:
                    raise RuntimeError(
                        f"No kept token to assign for victim {victim} at position {pos}."
                    )

                if merge_metric == "k":
                    sims = torch.matmul(k_head[kept_idx], k_head[victim])
                else:
                    sims = torch.matmul(v_head[kept_idx], v_head[victim])
                y = int(kept_idx[torch.argmax(sims)].item())

                parent[victim] = y
                count[y] += count[victim]
                count[victim] = 0

            invisible_now = ~in_cache[:total_available]
            mask[out_h, i, :total_available][invisible_now] = float("-inf")

            belong[out_h, i, :total_available] = parent[:total_available]
            count_mtx[out_h, i, :total_available] = count[:total_available]

    t2 = time.time()
    print(f"[h2o_with_belong_all] mask+belong build time: {t2 - t1:.4f}s")

    for i, pos in enumerate(pos_list):
        mask[:, i, pos + 1 :] = float("-inf")
        belong[:, i, pos + 1 :] = -1

    return mask, belong, count_mtx


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
