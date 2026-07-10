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
    collect_stats=False,
    full_attention_stats=None,
):
    layer_to_patch = {}
    stats_by_layer = {}
    aggregate_stats = {}
    for layer_idx in layer_idx_list:
        if layer_idx < full_attention_layers:
            if collect_stats and full_attention_stats is not None:
                layer_stats = dict(full_attention_stats)
                stats_by_layer[int(layer_idx)] = layer_stats
                merge_stats(aggregate_stats, layer_stats)
            continue
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        patch_result = build_patch(layer_ctx, layer_idx, artifacts)
        if collect_stats:
            patch_hidden, layer_stats = patch_result
            stats_by_layer[int(layer_idx)] = layer_stats
            merge_stats(aggregate_stats, layer_stats)
        else:
            patch_hidden = patch_result
        layer_to_patch[layer_idx] = patch_hidden
        del artifacts, layer_ctx
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    if layer_to_patch:
        logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=layer_to_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        if collect_stats:
            return logits, summarize_stats(aggregate_stats, stats_by_layer)
        return logits

    with torch.no_grad():
        logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    if collect_stats:
        return logits, summarize_stats(aggregate_stats, stats_by_layer)
    return logits


def merge_stats(total, add):
    for key, value in add.items():
        total[key] = int(total.get(key, 0)) + int(value)


def summarize_stats(aggregate_stats, stats_by_layer):
    total_available = max(int(aggregate_stats.get("total_available", 0)), 1)
    rows = max(int(aggregate_stats.get("rows", 0)), 1)
    hybrid_tokens = int(aggregate_stats.get("hybrid_tokens", 0))
    aggregate = {
        **aggregate_stats,
        "mean_hybrid_tokens": float(hybrid_tokens / rows),
        "mean_budget_causal": float(hybrid_tokens / total_available),
        "mean_budget_visible": float(hybrid_tokens / total_available),
    }
    by_layer = {}
    for layer_idx, stats in stats_by_layer.items():
        layer_total = max(int(stats.get("total_available", 0)), 1)
        layer_rows = max(int(stats.get("rows", 0)), 1)
        layer_hybrid = int(stats.get("hybrid_tokens", 0))
        by_layer[layer_idx] = {
            **stats,
            "mean_hybrid_tokens": float(layer_hybrid / layer_rows),
            "mean_budget_causal": float(layer_hybrid / layer_total),
            "mean_budget_visible": float(layer_hybrid / layer_total),
        }
    return {"aggregate": aggregate, "by_layer": by_layer}
