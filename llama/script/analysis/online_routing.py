from types import SimpleNamespace

import torch
from transformers.models.llama import modeling_llama


def run_with_multilayer_patches(ctx, layer_to_patch, pos_list, model_inputs):
    if not layer_to_patch:
        raise ValueError("layer_to_patch is empty")

    pos_idx = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    handles = []

    def hook_factory(layer_idx):
        patch_hidden = layer_to_patch[layer_idx]

        def _hook(_module, _module_inputs, module_output):
            if isinstance(module_output, tuple):
                attn_out = module_output[0].clone()
                attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
                return (attn_out,) + module_output[1:]

            attn_out = module_output.clone()
            attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
            return attn_out

        return _hook

    for layer_idx in layer_to_patch.keys():
        layer = ctx.model.model.layers[layer_idx]
        handle = layer.self_attn.register_forward_hook(hook_factory(layer_idx))
        handles.append(handle)

    try:
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    finally:
        for handle in handles:
            handle.remove()

    return logits


def capture_layer_artifacts(ctx, layer_idx, pos_list, model_inputs, layer_to_patch=None):
    """Run one forward pass and capture Q/K/V + attention output for a specific layer."""
    if layer_to_patch is None:
        layer_to_patch = {}

    pos_idx = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    target_layer_input = {}
    captured = {}
    handles = []

    def layer_input_hook(_module, inp, _out):
        target_layer_input["hidden"] = inp[0].detach().cpu()

    def patch_hook_factory(patch_hidden):
        def _hook(_module, _module_inputs, module_output):
            if isinstance(module_output, tuple):
                attn_out = module_output[0].clone()
                attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
                return (attn_out,) + module_output[1:]

            attn_out = module_output.clone()
            attn_out[:, pos_idx, :] = patch_hidden.unsqueeze(0).to(attn_out.dtype)
            return attn_out

        return _hook

    for patch_layer_idx, patch_hidden in layer_to_patch.items():
        layer = ctx.model.model.layers[patch_layer_idx]
        handle = layer.self_attn.register_forward_hook(patch_hook_factory(patch_hidden))
        handles.append(handle)

    target_layer = ctx.model.model.layers[layer_idx]
    handles.append(target_layer.register_forward_hook(layer_input_hook))

    original_eager = modeling_llama.eager_attention_forward

    # Capture raw Q/K/V and pre-o_proj attention output from the current forward state.
    def eager_wrapper(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        attn_output, attn_weights = original_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        module_layer_idx = getattr(module, "layer_idx", None)
        if module_layer_idx == layer_idx:
            captured["q"] = query.detach().cpu()
            captured["k"] = key.detach().cpu()
            captured["v"] = value.detach().cpu()
            captured["attn_output"] = attn_output.detach().cpu()
        return attn_output, attn_weights

    modeling_llama.eager_attention_forward = eager_wrapper
    try:
        with torch.no_grad():
            _ = ctx.model(**model_inputs, use_cache=False)
    finally:
        modeling_llama.eager_attention_forward = original_eager
        for handle in handles:
            handle.remove()

    if "q" not in captured or "hidden" not in target_layer_input:
        raise RuntimeError(f"Failed to capture runtime artifacts for layer {layer_idx}.")

    return {
        "q": captured["q"],
        "k": captured["k"],
        "v": captured["v"],
        "attn_output": captured["attn_output"],
        "layer_input": target_layer_input["hidden"],
    }


def build_runtime_layer_ctx(base_ctx, layer_idx, artifacts):
    """Build a minimal context object so existing optimize/mask utilities can be reused."""
    return SimpleNamespace(
        model=base_ctx.model,
        tokenizer=base_ctx.tokenizer,
        rope_qkv={
            layer_idx: {
                "q": artifacts["q"],
                "k": artifacts["k"],
                "v": artifacts["v"],
            }
        },
        inputs=base_ctx.inputs,
        outputs=None,
        attn_output={
            layer_idx: {
                "output": artifacts["attn_output"],
            }
        },
        layer_input={layer_idx: artifacts["layer_input"]},
        gt_label=base_ctx.gt_label,
        model_config=base_ctx.model_config,
        dtype=base_ctx.dtype,
        device=base_ctx.device,
    )
