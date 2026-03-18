import torch
from torch.nn import functional as F


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

    layer = ctx.model.model.layers[layer_idx]
    original = ctx.attn_output[layer_idx]["output"][0, pos_list].permute(1, 0, 2).to(device)
    V_head = ctx.rope_qkv[layer_idx]["v"].to(device)[0][head_idx]

    V_new = alpha.to(device) @ V_head.float()
    output = original.clone()
    output[head_idx] = V_new.to(V_head.dtype)

    attn_hidden = layer.self_attn.o_proj(output.permute(1, 0, 2).reshape(len(pos_list), -1))
    return attn_hidden


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

