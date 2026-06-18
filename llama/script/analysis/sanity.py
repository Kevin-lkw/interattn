"""
This module contains utility functions
comput NLL and KL
"""
import torch
from torch.nn import functional as F


def expand_kv_to_query_heads(kv, num_query_heads, model_config=None):
    """Expand GQA K/V heads to query-head layout when needed."""
    num_kv_heads = kv.shape[0]
    if num_kv_heads == num_query_heads:
        return kv
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot map {num_kv_heads} K/V heads to {num_query_heads} query heads."
        )
    if model_config is not None:
        config_kv_heads = getattr(model_config, "num_key_value_heads", num_kv_heads)
        if int(config_kv_heads) != int(num_kv_heads):
            raise ValueError(
                "Captured K/V head count does not match model config: "
                f"captured={num_kv_heads}, config={config_kv_heads}."
            )
    return kv.repeat_interleave(num_query_heads // num_kv_heads, dim=0)


def kv_group_size(num_query_heads, num_kv_heads):
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot map {num_kv_heads} K/V heads to {num_query_heads} query heads."
        )
    return num_query_heads // num_kv_heads


def kv_head_indices_for_query_heads(head_idx, model_config=None, num_kv_heads=None):
    if isinstance(head_idx, int):
        head_idx = [head_idx]
    if model_config is None:
        if num_kv_heads is None:
            return list(head_idx)
        num_query_heads = max(head_idx) + 1 if head_idx else num_kv_heads
    else:
        num_query_heads = int(getattr(model_config, "num_attention_heads"))
        if num_kv_heads is None:
            num_kv_heads = int(getattr(model_config, "num_key_value_heads", num_query_heads))
    group_size = kv_group_size(num_query_heads, int(num_kv_heads))
    return [int(head) // group_size for head in head_idx]


def grouped_query_heads(head_idx, model_config=None, num_kv_heads=None):
    if isinstance(head_idx, int):
        head_idx = [head_idx]
    kv_indices = kv_head_indices_for_query_heads(head_idx, model_config, num_kv_heads)
    groups = []
    current = {}
    for out_idx, (query_head, kv_head) in enumerate(zip(head_idx, kv_indices)):
        current.setdefault(int(kv_head), {"out_indices": [], "query_heads": []})
        current[int(kv_head)]["out_indices"].append(out_idx)
        current[int(kv_head)]["query_heads"].append(int(query_head))
    for kv_head in sorted(current):
        groups.append(
            (
                kv_head,
                current[kv_head]["out_indices"],
                current[kv_head]["query_heads"],
            )
        )
    return groups


def select_kv_for_query_heads(kv, head_idx, model_config=None):
    kv_indices = kv_head_indices_for_query_heads(
        head_idx,
        model_config,
        num_kv_heads=kv.shape[0],
    )
    return kv[kv_indices]

def compute_metrics(ref_tail_logits, student_tail_logits, labels, unbiased=False):
    # [*, vocab]
    p_teacher = F.softmax(ref_tail_logits, dim=-1)
    logp_teacher = F.log_softmax(ref_tail_logits, dim=-1)
    logp_student = F.log_softmax(student_tail_logits, dim=-1)

    # 逐 token KL: shape = la bels.shape
    kl_per_token = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)

    # 逐 token NLL
    teacher_nll_per_token = F.cross_entropy(
        ref_tail_logits.reshape(-1, ref_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    student_nll_per_token = F.cross_entropy(
        student_tail_logits.reshape(-1, student_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    nll_gap_per_token = student_nll_per_token - teacher_nll_per_token

    return {
        "sanity_kl": kl_per_token.mean().item(),
        "sanity_kl_std": kl_per_token.std(unbiased=unbiased).item(),
        "teacher_nll": teacher_nll_per_token.mean().item(),
        "teacher_nll_std": teacher_nll_per_token.std(unbiased=unbiased).item(),
        "student_nll": student_nll_per_token.mean().item(),
        "student_nll_std": student_nll_per_token.std(unbiased=unbiased).item(),
        "nll_gap": nll_gap_per_token.mean().item(),
        "nll_gap_std": nll_gap_per_token.std(unbiased=unbiased).item(),
    }


def move_model_inputs_to_device(raw_inputs, device):
    if isinstance(raw_inputs, dict):
        moved = {}
        for key, value in raw_inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved
    if torch.is_tensor(raw_inputs):
        return {"input_ids": raw_inputs.to(device)}
    raise TypeError(f"Unsupported input type for model forward: {type(raw_inputs)}")


def build_modified_attn_hidden(ctx, layer_idx, head_idx, pos_list, alpha, device=None):
    if device is None:
        device = ctx.device

    if isinstance(head_idx, int):
        head_idx = [head_idx]

    with torch.no_grad():
        layer = ctx.model.model.layers[layer_idx]
        original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)
        V_head = select_kv_for_query_heads(
            ctx.rope_qkv[layer_idx]["v"].to(device)[0],
            head_idx,
            ctx.model_config,
        )

        V_new = alpha.detach().to(device) @ V_head.float()
        output = original.clone()
        output[head_idx] = V_new.to(V_head.dtype)

        attn_hidden = layer.self_attn.o_proj(output.permute(1, 0, 2).reshape(len(pos_list), -1))
        return attn_hidden.detach()


def get_tail_labels(ctx, pos_list, device):
    labels = ctx.gt_label
    if not torch.is_tensor(labels):
        raise TypeError(f"Unsupported gt_label type: {type(labels)}")
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    return labels[:, pos_list].to(device)


def compute_final_kl_with_reinjected_alpha(ctx, layer_idx, pos_list, attn_hidden_patch, model_inputs, ref_tail_logits):
    layer = ctx.model.model.layers[layer_idx]
    pos_idx = torch.tensor(pos_list, device=attn_hidden_patch.device, dtype=torch.long)

    def _hook(_module, _module_inputs, module_output):
        if isinstance(module_output, tuple):
            attn_out = module_output[0].clone()
            attn_out[:, pos_idx, :] = attn_hidden_patch.unsqueeze(0).to(attn_out.dtype)
            return (attn_out,) + module_output[1:]

        attn_out = module_output.clone()
        attn_out[:, pos_idx, :] = attn_hidden_patch.unsqueeze(0).to(attn_out.dtype)
        return attn_out

    handle = layer.self_attn.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    finally:
        handle.remove()

    p_teacher = F.softmax(ref_tail_logits, dim=-1)
    logp_teacher = F.log_softmax(ref_tail_logits, dim=-1)
    logp_student = F.log_softmax(logits, dim=-1)
    kl = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1).mean().item()

    labels = get_tail_labels(ctx, pos_list, logits.device)
    teacher_nll = F.cross_entropy(
        ref_tail_logits.reshape(-1, ref_tail_logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    ).item()
    student_nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    ).item()

    return {
        "sanity_kl": kl,
        "teacher_nll": teacher_nll,
        "student_nll": student_nll,
        "nll_gap": student_nll - teacher_nll,
    }


def has_full_sanity_metrics(entry):
    required_keys = ["sanity_kl", "teacher_nll", "student_nll", "nll_gap"]
    return all(key in entry for key in required_keys)
