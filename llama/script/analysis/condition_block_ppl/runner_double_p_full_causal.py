"""Full-causal Double-P reference for fixed-chunk perplexity.

The published Double-P algorithm clusters a completed prompt and sparsifies
subsequent decode queries.  Fixed-chunk PPL has no distinguished prompt, so
this extension applies Double-P to every scored query without a fixed dense
prefill.  Clusters are rebuilt causally whenever another complete
``cluster_size`` group has left the exact local window.  Tokens that have left
the window but do not yet form a complete group remain exact until the next
reclustering point.

Every cluster is therefore constructed only from keys at or before the query
positions that consume it.  The implementation is an eager PyTorch accuracy
reference, not a latency implementation.
"""

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from ..condition_block_gen.methods.double_p import (
    _double_p_decode_output,
    build_double_p_prompt_clusters,
)
from ..online_routing import capture_layer_artifacts, run_with_multilayer_patches
from .runner_cond_block import (
    _full_attention_stats,
    _merge_stats,
    _summarize_budget,
)
from .runner_double_p import _setting_key


def causal_clustered_end(
    position: int,
    *,
    cluster_size: int,
    sink_tokens: int,
    window_size: int,
) -> int:
    """Return the exclusive prefix end eligible for causal clustering.

    A zero return value means that no complete cluster is available yet.  A
    positive value includes the exact sink prefix followed by an integer
    number of complete ``cluster_size`` groups.
    """

    visible = int(position) + 1
    clusterable = max(visible - int(sink_tokens) - int(window_size), 0)
    complete = (clusterable // int(cluster_size)) * int(cluster_size)
    if complete == 0:
        return 0
    return int(sink_tokens) + complete


def causal_epoch_groups(
    pos_list,
    *,
    cluster_size: int,
    sink_tokens: int,
    window_size: int,
):
    groups = {}
    for output_idx, position in enumerate(pos_list):
        clustered_end = causal_clustered_end(
            position,
            cluster_size=cluster_size,
            sink_tokens=sink_tokens,
            window_size=window_size,
        )
        groups.setdefault(clustered_end, []).append((output_idx, int(position)))
    return groups


def full_causal_double_p_attention(
    *,
    q_all: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    pos_list,
    cluster_size: int,
    kmeans_iters: int,
    p1: float,
    p2: float,
    sink_tokens: int,
    window_size: int,
):
    """Apply causally rebuilt Double-P clusters to every requested query."""

    if q_all.ndim != 3 or k_all.ndim != 3 or v_all.ndim != 3:
        raise ValueError("q_all, k_all, and v_all must be rank-three tensors")
    if k_all.shape != v_all.shape:
        raise ValueError("k_all and v_all must have identical shapes")
    n_heads, seq_len, head_dim = map(int, q_all.shape)
    n_kv_heads = int(k_all.shape[0])
    if int(k_all.shape[1]) != seq_len or int(k_all.shape[2]) != head_dim:
        raise ValueError("Q/K/V sequence length and head dimension must match")
    if n_heads % n_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")
    if int(cluster_size) <= 0 or int(kmeans_iters) <= 0:
        raise ValueError("cluster_size and kmeans_iters must be positive")
    if int(sink_tokens) < 0 or int(window_size) < 0:
        raise ValueError("sink_tokens and window_size must be non-negative")
    positions = [int(position) for position in pos_list]
    if any(position < 0 or position >= seq_len for position in positions):
        raise ValueError("pos_list contains a position outside the Q/K/V sequence")

    group_size = n_heads // n_kv_heads
    output = torch.empty(
        n_heads,
        len(positions),
        head_dim,
        device=q_all.device,
        dtype=torch.float32,
    )
    aggregate_stats = {}
    groups = causal_epoch_groups(
        positions,
        cluster_size=cluster_size,
        sink_tokens=sink_tokens,
        window_size=window_size,
    )
    for clustered_end, entries in groups.items():
        output_indices = [entry[0] for entry in entries]
        epoch_positions = [entry[1] for entry in entries]
        pos_tensor = torch.tensor(
            epoch_positions,
            device=q_all.device,
            dtype=torch.long,
        )
        q_grouped = q_all[:, pos_tensor].reshape(
            n_kv_heads,
            group_size,
            len(epoch_positions),
            head_dim,
        )
        prompt_clusters = build_double_p_prompt_clusters(
            k_all[:, :clustered_end],
            v_all[:, :clustered_end],
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
            sink_tokens=min(int(sink_tokens), int(clustered_end)),
            window_size=0,
        )
        epoch_output, epoch_stats = _double_p_decode_output(
            q_grouped=q_grouped,
            k_all=k_all,
            v_all=v_all,
            prompt_clusters=prompt_clusters,
            pos_tensor=pos_tensor,
            p1=p1,
            p2=p2,
        )
        output[:, output_indices] = epoch_output.reshape(
            n_heads,
            len(epoch_positions),
            head_dim,
        )
        _merge_stats(aggregate_stats, epoch_stats)
    return output, aggregate_stats


def build_full_causal_double_p_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    cluster_size,
    kmeans_iters,
    p1,
    p2,
    sink_tokens,
    window_size,
):
    if int(artifacts["q"].shape[0]) != 1:
        raise ValueError("Full-causal Double-P expects batch_size=1")
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    output, stats = full_causal_double_p_attention(
        q_all=q_all,
        k_all=k_all,
        v_all=v_all,
        pos_list=pos_list,
        cluster_size=cluster_size,
        kmeans_iters=kmeans_iters,
        p1=p1,
        p2=p2,
        sink_tokens=sink_tokens,
        window_size=window_size,
    )

    output_dtype = artifacts["attn_output"].dtype
    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.permute(1, 0, 2)
        .reshape(len(pos_list), -1)
        .to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach().to(output_dtype), stats


def run_full_causal_setting(
    ctx,
    args,
    p1,
    p2,
    layer_idx_list,
    pos_list,
    model_inputs,
):
    layer_to_patch = {}
    budget_stats = {}
    aggregate_stats = {}
    setting = _setting_key(p1, p2)

    if float(p1) == 1.0 and float(p2) == 1.0:
        for layer_idx in layer_idx_list:
            layer_stats = _full_attention_stats(
                n_heads=ctx.model_config.num_attention_heads,
                pos_list=pos_list,
            )
            budget_stats[int(layer_idx)] = _summarize_budget(
                layer_stats,
                seq_len=args.seq_len,
            )
            _merge_stats(aggregate_stats, layer_stats)
        with torch.no_grad():
            logits = ctx.model(
                **model_inputs,
                use_cache=False,
            ).logits[:, pos_list, :].float()
        return logits, {}, {
            "aggregate": _summarize_budget(aggregate_stats, seq_len=args.seq_len),
            "by_layer": budget_stats,
        }

    layer_iter = tqdm(
        layer_idx_list,
        desc=f"full_causal_{setting}",
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        t0 = time.time()
        if layer_idx < args.full_attention_layers:
            layer_stats = _full_attention_stats(
                n_heads=ctx.model_config.num_attention_heads,
                pos_list=pos_list,
            )
        else:
            artifacts = capture_layer_artifacts(
                ctx=ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                model_inputs=model_inputs,
                layer_to_patch=layer_to_patch,
            )
            patch_hidden, layer_stats = build_full_causal_double_p_patch(
                ctx=ctx,
                layer_idx=layer_idx,
                artifacts=artifacts,
                pos_list=pos_list,
                cluster_size=args.cluster_size,
                kmeans_iters=args.kmeans_iters,
                p1=p1,
                p2=p2,
                sink_tokens=args.sink_tokens,
                window_size=args.window_size,
            )
            layer_to_patch[layer_idx] = patch_hidden
            del artifacts

        budget_stats[int(layer_idx)] = _summarize_budget(
            layer_stats,
            seq_len=args.seq_len,
        )
        _merge_stats(aggregate_stats, layer_stats)
        layer_iter.set_postfix(
            layer=int(layer_idx),
            budget=f"{budget_stats[int(layer_idx)]['mean_budget_causal']:.3f}",
            seconds=f"{time.time() - t0:.1f}",
        )
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    logits = run_with_multilayer_patches(
        ctx=ctx,
        layer_to_patch=layer_to_patch,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )
    return logits, layer_to_patch, {
        "aggregate": _summarize_budget(aggregate_stats, seq_len=args.seq_len),
        "by_layer": budget_stats,
    }
