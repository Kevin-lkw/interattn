"""
Inspect cluster-level conditioning terms for simple contiguous block clusters.

For budget x, this script uses block_size = round(1 / x).  At each query q,
visible keys [0, q] are partitioned into contiguous blocks:

    [0, block_size), [block_size, 2 * block_size), ...

The first token in each block is used as the cluster root id.  The condition
statistics and plots are shared with condition.py.
"""

import argparse
import os

import torch

from .condition import (
    _align_prefix_patches_to_pos_list,
    _choose_evenly,
    _resolve_query_positions,
    _build_routing_pos_list,
    collect_condition_stats,
)
from .compare_utils import (
    add_common_compare_args,
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
    resolve_output_dir,
    validate_common_args,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot condition stats for contiguous block clusters."
    )
    add_common_compare_args(
        parser,
        strategy_choices=["block"],
        default_strategy="block",
        include_loss_type=True,
        include_plot_dpi=True,
        prefix_mode_default="full_attention",
        prefix_mode_choices=("full_attention", "optimal_saved", "baseline_rebuild"),
        prefix_mode_help=(
            "How to prepare layers before target layer. "
            "full_attention: keep previous layers unpatched; "
            "optimal_saved: load saved optimal patch_hidden for layers < target; "
            "baseline_rebuild: rebuild block-cluster baseline patches online for layers < target."
        ),
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=256,
        help="Inclusive start position for query selection.",
    )
    parser.add_argument(
        "--query-end",
        type=int,
        default=None,
        help="Exclusive end position for query selection. Defaults to --pos-end/--seq-len.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=16,
        help="Number of query positions to plot per head.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        default=None,
        help="Explicit query positions. Overrides --query-start/--query-end/--num-queries.",
    )
    parser.add_argument(
        "--sample-heads",
        type=int,
        default=4,
        help="When --head/--heads is not set, choose this many heads evenly across all heads.",
    )
    parser.add_argument(
        "--cluster-order",
        choices=["p_desc", "p_full_desc", "size_desc", "root"],
        default="root",
        help="Order clusters on the x-axis and in the saved/printed table.",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=200,
        help="Print at most this many TSV rows to stdout. Use -1 to print all rows.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the default block size round(1 / budget).",
    )
    parser.add_argument(
        "--condition-eps",
        type=float,
        nargs="+",
        default=[5.0, 1.0, 0.1],
        help="Thresholds for counting high-condition clusters and hybrid full-attention budget.",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="exact",
        help=(
            "How to compute delta_C. exact uses all q dot K_i scores; "
            "range_bound uses per-dimension K min/max to upper-bound delta_C."
        ),
    )
    return parser.parse_args()


def _resolve_block_size(args):
    if args.block_size is not None:
        if args.block_size <= 0:
            raise ValueError("--block-size must be > 0")
        return int(args.block_size)
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    return max(1, int(round(1.0 / args.budget)))


def _build_block_belong_and_mask(*, device, head_idx, pos_list, seq_len, block_size):
    n_heads = len(head_idx)
    n_pos = len(pos_list)
    belong_root = torch.full(
        (n_heads, n_pos, seq_len),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    route_mask = torch.full(
        (n_heads, n_pos, seq_len),
        fill_value=float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    token_idx = torch.arange(seq_len, device=device, dtype=torch.long)
    roots_all = (token_idx // block_size) * block_size

    for i, pos in enumerate(pos_list):
        total_available = int(pos) + 1
        row_roots = roots_all[:total_available]
        belong_root[:, i, :total_available] = row_roots.unsqueeze(0)

        kept = torch.unique(row_roots)
        route_mask[:, i, kept] = 0.0
        route_mask[:, i, pos + 1 :] = float("-inf")

    return route_mask, belong_root


def _prepare_prefix_patches(args, ctx, routing_pos_list, model_inputs, block_size):
    if args.prefix_mode == "full_attention":
        return {}
    if args.prefix_mode == "optimal_saved":
        prefix_patches = build_optimal_saved_prefix_patches(
            args=args,
            target_layer=args.layer,
            budget=args.budget,
            device=ctx.device,
        )
        return _align_prefix_patches_to_pos_list(prefix_patches, routing_pos_list, args.seq_len)

    return build_baseline_prefix_patches(
        ctx=ctx,
        args=args,
        target_layer=args.layer,
        pos_list=routing_pos_list,
        model_inputs=model_inputs,
        build_mask_fn=lambda layer_ctx, _layer_idx, hi: _build_block_belong_and_mask(
            device=layer_ctx.device,
            head_idx=hi,
            pos_list=routing_pos_list,
            seq_len=args.seq_len,
            block_size=block_size,
        )[0],
    )


def main():
    set_seed(42)
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.print_rows < -1:
        raise ValueError("--print-rows must be >= -1")

    block_size = _resolve_block_size(args)
    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_common_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )

    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    if args.head is None and args.heads is None:
        head_idx = _choose_evenly(head_idx, args.sample_heads)

    routing_pos_list = _build_routing_pos_list(args)
    query_positions = _resolve_query_positions(args, routing_pos_list)
    print(
        f"Build block clusters with block_size={block_size}, "
        f"pos_list=[{routing_pos_list[0]}, {routing_pos_list[-1]}] "
        f"({len(routing_pos_list)} positions); condition queries={query_positions}"
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    output_dir = resolve_output_dir(
        args=args,
        head_idx=head_idx,
        compare_tag="condition_block",
        include_loss_type=True,
    )

    prefix_patches = _prepare_prefix_patches(
        args=args,
        ctx=ctx,
        routing_pos_list=routing_pos_list,
        model_inputs=model_inputs,
        block_size=block_size,
    )
    print("Prefix patches prepared for layers", list(prefix_patches.keys()))
    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=args.layer,
        pos_list=routing_pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, args.layer, artifacts)

    route_mask, belong_root = _build_block_belong_and_mask(
        device=ctx.device,
        head_idx=head_idx,
        pos_list=routing_pos_list,
        seq_len=args.seq_len,
        block_size=block_size,
    )

    result = collect_condition_stats(
        out_dir=output_dir,
        layer_ctx=layer_ctx,
        route_mask=route_mask,
        belong_root=belong_root,
        pos_list=routing_pos_list,
        head_labels=head_idx,
        query_positions=query_positions,
        args=args,
    )

    summary_path = os.path.join(output_dir, "condition_block_summary.pt")
    torch.save(
        {
            "config": vars(args),
            "block_size": int(block_size),
            "layer": int(args.layer),
            "heads": head_idx,
            "pos_list": routing_pos_list,
            "query_positions": query_positions,
            "outputs": result,
            "belong_root": belong_root.detach().cpu(),
        },
        summary_path,
    )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
