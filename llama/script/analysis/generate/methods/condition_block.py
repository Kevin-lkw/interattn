import math

import torch

from ...runner_cond_block import _resolve_block_size
from ...sanity import grouped_query_heads
from .patching import run_with_prefill_only_patches


def run_prefill_only_condition_block(
    *,
    ctx,
    args,
    eps,
    layer_idx_list,
    prompt_len,
    pos_list,
    model_inputs,
):
    if math.isclose(float(args.budget), 1.0):
        with torch.no_grad():
            return ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()

    def _build_patch(layer_ctx, layer_idx, artifacts):
        return build_prefill_only_condition_block_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            prompt_len=prompt_len,
            block_size=args.block_size,
            eps=eps,
            delta_mode=args.delta_mode,
        )

    return run_with_prefill_only_patches(
        ctx=ctx,
        layer_idx_list=layer_idx_list,
        pos_list=pos_list,
        model_inputs=model_inputs,
        full_attention_layers=args.full_attention_layers,
        build_patch=_build_patch,
    )


def build_condition_args(method, prompt_len):
    from types import SimpleNamespace

    args = SimpleNamespace(
        seq_len=prompt_len,
        budget=method.budget,
        block_size=method.condition_block_size,
        full_attention_layers=method.full_attention_layers,
        delta_mode=method.condition_delta_mode,
    )
    args.block_size = _resolve_block_size(args)
    return args


def build_prefill_only_condition_block_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    prompt_len,
    block_size,
    eps,
    delta_mode,
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
            head_conditions = []
            for head_offset, query_head in enumerate(query_heads):
                head_conditions.append(
                    condition_tensor_for_prompt_blocks(
                        q_all[query_head, pos].float(),
                        k_group[head_offset, :prefix_len],
                        v_group[head_offset, :prefix_len],
                        block_size,
                        delta_mode,
                    )
                )
            selected_blocks = torch.stack(head_conditions, dim=0).mean(dim=0) > eps

            for head_offset, query_head in enumerate(query_heads):
                q = q_all[query_head, pos].float()
                logits_parts = []
                value_parts = []
                block_idx = 0
                for start in range(0, prefix_len, block_size):
                    end = min(start + block_size, prefix_len)
                    k_block = k_group[head_offset, start:end]
                    v_block = v_group[head_offset, start:end]
                    if bool(selected_blocks[block_idx].item()):
                        logits_parts.append(torch.mv(k_block, q) / math.sqrt(q.numel()))
                        value_parts.append(v_block)
                    else:
                        k_bar = k_block.mean(dim=0)
                        v_bar = v_block.mean(dim=0, keepdim=True)
                        score = torch.dot(q, k_bar) / math.sqrt(q.numel())
                        logits_parts.append((math.log(end - start) + score).reshape(1))
                        value_parts.append(v_bar)
                    block_idx += 1

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


def condition_tensor_for_prompt_blocks(q, k_head, v_head, block_size, delta_mode):
    scale = math.sqrt(q.numel())
    seq_len = k_head.shape[0]
    s_vals = []
    deltas = []
    b_cs = []
    size_vals = []
    b_all = torch.norm(v_head.float(), p=2, dim=-1).max()

    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        k_cluster = k_head[start:end].float()
        v_cluster = v_head[start:end].float()
        k_bar = k_cluster.mean(dim=0)
        s_c = torch.dot(q.float(), k_bar.float()) / scale
        if delta_mode == "exact":
            qk_cluster = torch.mv(k_cluster, q.float()) / scale
            delta = (qk_cluster - s_c).abs().max()
        elif delta_mode == "range_bound":
            k_max = k_cluster.max(dim=0).values
            k_min = k_cluster.min(dim=0).values
            upper_score = torch.maximum(q.float() * k_max, q.float() * k_min).sum() / scale
            lower_score = torch.minimum(q.float() * k_max, q.float() * k_min).sum() / scale
            delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        else:
            raise ValueError(f"Unknown delta mode: {delta_mode}")
        s_vals.append(s_c)
        deltas.append(delta)
        b_cs.append(torch.norm(v_cluster, p=2, dim=-1).max())
        size_vals.append(float(end - start))

    s_tensor = torch.stack(s_vals).float()
    size_tensor = torch.tensor(size_vals, device=s_tensor.device, dtype=torch.float32)
    delta_tensor = torch.stack(deltas).float()
    b_c_tensor = torch.stack(b_cs).float()
    p_tensor = torch.softmax(torch.log(size_tensor) + s_tensor, dim=0)
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    return p_tensor * (
        2.0 * b_all * (torch.cosh(delta_tensor) - 1.0) / denom
        + 2.0 * b_c_tensor * torch.tanh(delta_tensor / 2.0)
    )
