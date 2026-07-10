import math

import torch
from torch.nn import functional as F

from ...sanity import build_modified_attn_hidden, grouped_query_heads
from .patching import run_with_prefill_only_patches


def run_prefill_only_attention_topk(
    *,
    ctx,
    budget,
    full_attention_layers,
    seq_len,
    prompt_len,
    pos_list,
    model_inputs,
):
    if math.isclose(float(budget), 1.0):
        with torch.no_grad():
            return ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()

    def _build_patch(layer_ctx, layer_idx, artifacts):
        alpha = build_prefill_only_attention_topk_alpha(
            artifacts=artifacts,
            pos_list=pos_list,
            budget=budget,
            seq_len=seq_len,
            prompt_len=prompt_len,
            device=ctx.device,
            model_config=ctx.model_config,
        )
        return build_modified_attn_hidden(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            head_idx=list(range(ctx.model_config.num_attention_heads)),
            pos_list=pos_list,
            alpha=alpha,
            device=ctx.device,
        )

    return run_with_prefill_only_patches(
        ctx=ctx,
        layer_idx_list=list(range(ctx.model_config.num_hidden_layers)),
        pos_list=pos_list,
        model_inputs=model_inputs,
        full_attention_layers=full_attention_layers,
        build_patch=_build_patch,
    )


def build_prefill_only_attention_topk_alpha(
    *,
    artifacts,
    pos_list,
    budget,
    seq_len,
    prompt_len,
    device,
    model_config,
):
    q_all = artifacts["q"].to(device)[0].float()
    k_all = artifacts["k"].to(device)[0].float()
    pos_tensor = torch.tensor(pos_list, device=device, dtype=torch.long)
    q_pos = q_all[:, pos_tensor, :]
    scale = math.sqrt(q_all.shape[-1])
    qk_logits = torch.empty(
        q_all.shape[0],
        len(pos_list),
        seq_len,
        device=device,
        dtype=torch.float32,
    )
    groups = grouped_query_heads(
        list(range(q_all.shape[0])),
        model_config,
        num_kv_heads=k_all.shape[0],
    )
    for kv_head, out_indices, query_heads in groups:
        qk_logits[out_indices] = (
            torch.einsum("hqd,kd->hqk", q_pos[query_heads], k_all[kv_head]) / scale
        )

    key_idx = torch.arange(seq_len, device=device)
    causal = key_idx.view(1, 1, seq_len) <= pos_tensor.view(1, -1, 1)
    prefill_keys = key_idx < prompt_len
    decode_keys = key_idx >= prompt_len
    visible = max(1, int(prompt_len * budget))

    selected = (
        decode_keys.view(1, 1, seq_len) & causal
    ).expand(qk_logits.shape[0], -1, -1).clone()
    if visible >= prompt_len:
        selected |= (prefill_keys.view(1, 1, seq_len) & causal).expand_as(selected)
    else:
        for _kv_head, out_indices, _query_heads in groups:
            group_logits = qk_logits[out_indices]
            select_mask = causal[0] & prefill_keys.view(1, -1)
            select_logits = group_logits.mean(dim=0).masked_fill(
                ~select_mask,
                float("-inf"),
            )
            topk_idx = torch.topk(select_logits, k=visible, dim=-1, largest=True).indices
            group_selected = torch.zeros_like(select_logits, dtype=torch.bool)
            group_selected.scatter_(dim=-1, index=topk_idx, value=True)
            for out_idx in out_indices:
                selected[out_idx] |= group_selected
    selected &= causal

    alpha_logits = qk_logits.masked_fill(~selected, float("-inf"))
    return F.softmax(alpha_logits, dim=-1)
