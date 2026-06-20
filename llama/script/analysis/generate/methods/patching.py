import torch

from ...online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)


def run_with_prefill_only_patches(
    *,
    ctx,
    layer_idx_list,
    pos_list,
    model_inputs,
    full_attention_layers,
    build_patch,
):
    layer_to_patch = {}
    for layer_idx in layer_idx_list:
        if layer_idx < full_attention_layers:
            continue
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        layer_to_patch[layer_idx] = build_patch(layer_ctx, layer_idx, artifacts)
        del artifacts, layer_ctx
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    if layer_to_patch:
        return run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=layer_to_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )

    with torch.no_grad():
        return ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
