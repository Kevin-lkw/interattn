"""Shared loading/iteration for condition-bound experiments.

Run scripts in this folder as modules, e.g.

    python -m analysis.condition_block_ppl.condition_bound.hybrid_guarantee --device cuda:0

They reuse the analysis harness (model/dataset defaults of condition_block_corr:
Llama-2-7b-hf, wikitext, seq_len 1024, 4 heads, 16 queries, block_size round(1/budget)).
"""

import argparse
import os
import sys

_SCRIPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPT_ROOT)


def parse_args(description, default_layers=(10, 15, 20)):
    local = argparse.ArgumentParser(description=description)
    local.add_argument("--device", default="cuda:0")
    local.add_argument("--budget", type=float, default=0.1)
    local.add_argument("--layers", type=int, nargs="+", default=list(default_layers))
    opts = local.parse_args()

    from analysis.condition_block_ppl import condition_block_corr as cbc

    sys.argv = [sys.argv[0], "--budget", str(opts.budget),
                "--layer", str(opts.layers[0]), "--device", opts.device]
    args = cbc.parse_args()
    args.layers = opts.layers
    args.block_size = cbc._resolve_block_size(args)
    return args


def load_ctx(args):
    from analysis.runtime import load_context
    from analysis.runner_utils import set_seed, str_to_torch_dtype

    set_seed(42)
    ctx = load_context(args, dtype=str_to_torch_dtype(args.dtype), device=args.device)
    ctx.model.eval()
    return ctx


def layer_groups(ctx, args, layer):
    """Yield (q, k_head, v_head, query_pos) for every sampled head/query of a layer."""
    from analysis.condition_block_ppl.condition import (
        _build_routing_pos_list,
        _choose_evenly,
        _resolve_query_positions,
    )
    from analysis.experiment_utils import resolve_head_indices, validate_common_args
    from analysis.online_routing import build_runtime_layer_ctx, capture_layer_artifacts
    from analysis.sanity import grouped_query_heads, move_model_inputs_to_device

    args.layer = layer
    validate_common_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )
    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    if args.head is None and args.heads is None:
        head_idx = _choose_evenly(head_idx, args.sample_heads)
    pos_list = _build_routing_pos_list(args)
    query_positions = _resolve_query_positions(args, pos_list)
    inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    artifacts = capture_layer_artifacts(
        ctx=ctx, layer_idx=layer, pos_list=pos_list,
        model_inputs=inputs, layer_to_patch={},
    )
    lctx = build_runtime_layer_ctx(ctx, layer, artifacts)
    q_all = lctx.rope_qkv[layer]["q"].to(ctx.device)[0].float()
    k_all = lctx.rope_qkv[layer]["k"].to(ctx.device)[0].float()
    v_all = lctx.rope_qkv[layer]["v"].to(ctx.device)[0].float()
    for kv_head, _out, query_heads in grouped_query_heads(
        head_idx, ctx.model_config, num_kv_heads=k_all.shape[0],
    ):
        for qh in query_heads:
            for qp in query_positions:
                yield q_all[qh, int(qp)], k_all[kv_head], v_all[kv_head], int(qp)
