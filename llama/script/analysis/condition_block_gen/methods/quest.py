import math
from types import SimpleNamespace

import torch

from ...sanity import grouped_query_heads
from .patching import run_with_prefill_only_patches


def build_quest_args(method, prompt_len):
    requested_tokens = max(1, int(prompt_len * method.budget))
    page_budget = math.ceil(requested_tokens / method.quest_page_size)
    args = SimpleNamespace(seq_len=prompt_len, page_size=method.quest_page_size)
    return args, page_budget


def run_prefill_only_quest(
    *,
    ctx,
    args,
    budget,
    page_budget,
    layer_idx_list,
    prompt_len,
    pos_list,
    model_inputs,
):
    if math.isclose(float(budget), 1.0):
        with torch.no_grad():
            return ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()

    def _build_patch(layer_ctx, layer_idx, artifacts):
        return build_prefill_only_quest_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            prompt_len=prompt_len,
            page_size=args.page_size,
            page_budget=page_budget,
        )

    return run_with_prefill_only_patches(
        ctx=ctx,
        layer_idx_list=layer_idx_list,
        pos_list=pos_list,
        model_inputs=model_inputs,
        full_attention_layers=2,
        build_patch=_build_patch,
    )


def build_prefill_only_quest_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    prompt_len,
    page_size,
    page_budget,
):
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    output_dtype = artifacts["attn_output"].dtype

    n_heads = q_all.shape[0]
    n_pos = len(pos_list)
    pos_tensor = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    output = torch.empty(
        n_heads,
        n_pos,
        q_all.shape[-1],
        device=ctx.device,
        dtype=torch.float32,
    )

    for kv_head, _out_indices, query_heads in grouped_query_heads(
        list(range(n_heads)),
        ctx.model_config,
        num_kv_heads=k_all.shape[0],
    ):
        k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1).float()
        v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1).float()
        for local_pos, pos in enumerate(pos_tensor.tolist()):
            prefix_len = min(prompt_len, int(pos) + 1)
            suffix_start = prompt_len
            suffix_end = int(pos) + 1
            page_scores = []
            for start in range(0, prefix_len, page_size):
                end = min(start + page_size, prefix_len)
                k_page = k_group[:, start:end]
                k_max = k_page.max(dim=1).values
                k_min = k_page.min(dim=1).values
                q_page = q_all[query_heads, pos].float()
                score = torch.maximum(q_page * k_max, q_page * k_min).sum(dim=-1)
                page_scores.append(score)
            score_tensor = torch.stack(page_scores, dim=1)
            top_k = min(page_budget, score_tensor.shape[1])
            selected_pages = torch.zeros(score_tensor.shape[1], device=ctx.device, dtype=torch.bool)
            top_idx = torch.topk(score_tensor.mean(dim=0), k=top_k, largest=True).indices
            selected_pages.scatter_(dim=0, index=top_idx, value=True)

            for head_offset, query_head in enumerate(query_heads):
                q = q_all[query_head, pos].float()
                logits_parts = []
                value_parts = []
                page_idx = 0
                for start in range(0, prefix_len, page_size):
                    end = min(start + page_size, prefix_len)
                    if bool(selected_pages[page_idx].item()):
                        k_page = k_group[head_offset, start:end]
                        v_page = v_group[head_offset, start:end]
                        logits_parts.append(torch.mv(k_page, q) / math.sqrt(q.numel()))
                        value_parts.append(v_page)
                    page_idx += 1

                if suffix_end > suffix_start:
                    k_suffix = k_group[head_offset, suffix_start:suffix_end]
                    v_suffix = v_group[head_offset, suffix_start:suffix_end]
                    logits_parts.append(torch.mv(k_suffix, q) / math.sqrt(q.numel()))
                    value_parts.append(v_suffix)

                weights = torch.softmax(torch.cat(logits_parts, dim=0), dim=0)
                values = torch.cat(value_parts, dim=0)
                output[query_head, local_pos] = (weights.unsqueeze(-1) * values).sum(dim=0)

    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.to(output_dtype)
        .permute(1, 0, 2)
        .reshape(n_pos, -1)
        .to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach()
