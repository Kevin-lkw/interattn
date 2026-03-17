import math

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

        losses.append(loss.item())
        if step % 100 == 0:
            print("step", step, "loss:", loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    alpha = F.softmax(a_param + mask, dim=-1)
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


def gen_mask(
    ctx,
    layer_idx,
    pos_list,
    head_idx,
    strategy,
    budget,
    seq_len=4096,
):
    """
    Return mask for alpha_param, with shape [nh, n_pos, seq_len]
    """
    device = ctx.device
    mask = torch.zeros(len(pos_list), seq_len, device=device)

    visible = int(seq_len * budget)
    print(f"visible {visible} tokens for strategy {strategy} with budget {budget}")

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

        mask = torch.zeros(len(head_idx), len(pos_list), seq_len, device=device)
        for out_h, head in enumerate(head_idx):
            attention_score_head = attention_score[head]  # [seq_len, seq_len]
            accumulated_attention = torch.zeros(seq_len, device=device)

            for i, pos in enumerate(pos_list):
                total_available = pos + 1

                if total_available <= visible:
                    accumulated_attention += attention_score_head[pos]
                    continue

                cur_recent_budget = min(recent_budget, total_available)
                recent_start = total_available - cur_recent_budget
                recent_idx = torch.arange(recent_start, total_available, device=device)

                hh_candidate_end = recent_start
                cur_hh_budget = min(hh_budget, hh_candidate_end)

                keep = torch.zeros(total_available, dtype=torch.bool, device=device)
                keep[recent_idx] = True

                if cur_hh_budget > 0:
                    hh_scores = accumulated_attention[:hh_candidate_end]
                    topk_hh = torch.topk(hh_scores, k=cur_hh_budget, largest=True).indices
                    keep[topk_hh] = True

                mask[out_h, i, :total_available][~keep] = float("-inf")
                accumulated_attention += attention_score_head[pos]

    elif strategy == "kvmerger":
        raise NotImplementedError("KVMerger strategy is not implemented yet.")

    else:
        raise ValueError(f"Unknown strategy {strategy}")

    for i, pos in enumerate(pos_list):
        mask[:, i, pos + 1 :] = float("-inf")

    return mask  # [n_heads, n_pos, seq_len]
