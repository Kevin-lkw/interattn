"""
Check whether avgKV refinement gets closer to gt V on each layer.

This is a multi-layer checker version of compare_count_avgKV.py:
- For each target layer, optionally prepare prefix patches from
  optimal saved patches or online baseline rebuild.
- Compare baseline V-L2 and avgKV-refined V-L2 against gt V.
- Save per-layer summary for quick diagnosis.
"""

import argparse
import os

import torch

from .attention import gen_mask_h2o_with_belong
from .compare_count_avgKV import (
    build_avgk_count_refined_alpha,
    build_avgv_refined_v,
    get_qk_logits,
    v_l2_per_pos_from_v,
)
from .compare_utils import (
    build_baseline_prefix_patches,
    build_optimal_saved_prefix_patches,
    resolve_head_indices,
)
from .config import set_seed, str_to_torch_dtype
from .online_routing import build_runtime_layer_ctx, capture_layer_artifacts
from .runner import load_context, resolve_layers
from .sanity import move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check avgKV refinement quality across layers by comparing V-L2 to gt V."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument("--strategy", type=str, default="h2o", choices=["h2o"])
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--budget", type=float, required=True)

    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)

    parser.add_argument(
        "--loss-type",
        type=str,
        default="v_l2",
        choices=["logits_kl", "v_l2", "v_kl"],
        help="Used only when loading optimal_saved prefix patches.",
    )

    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="optimal_saved",
        choices=["optimal_saved", "baseline_rebuild"],
        help=(
            "How to prepare patches before each target layer. "
            "optimal_saved: load saved optimal patch_hidden for layers < target; "
            "baseline_rebuild: rebuild baseline patches online for layers < target."
        ),
    )

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")

    parser.add_argument("--pos-start", type=int, default=0)
    parser.add_argument("--pos-end", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args, num_layers, num_heads):
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.pos_start < 0:
        raise ValueError("--pos-start must be >= 0")
    if args.pos_end is not None and args.pos_end <= args.pos_start:
        raise ValueError("--pos-end must be larger than --pos-start")
    if args.head is not None and (args.head < 0 or args.head >= num_heads):
        raise ValueError(f"Invalid --head {args.head}; expected [0, {num_heads - 1}]")
    if args.heads is not None:
        for h in args.heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Invalid --heads entry {h}; expected [0, {num_heads - 1}]")

    if args.layers is not None:
        for layer in args.layers:
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Invalid --layers entry {layer}; expected [0, {num_layers - 1}]")


def resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        adaptive_str = "adaptive" if args.adaptive_budget else "fixed"
        out_dir = (
            f"../result/{args.dataset}_{args.start}/{adaptive_str}/{args.strategy}/"
            f"{args.loss_type}/check_avgkv_all_layers/budget_{args.budget:g}/prefix_{args.prefix_mode}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_prefix_patches(ctx, args, target_layer, pos_list, model_inputs):
    if args.prefix_mode == "optimal_saved":
        return build_optimal_saved_prefix_patches(
            args=args,
            target_layer=target_layer,
            budget=args.budget,
            device=ctx.device,
        )

    return build_baseline_prefix_patches(
        ctx=ctx,
        args=args,
        target_layer=target_layer,
        pos_list=pos_list,
        model_inputs=model_inputs,
        build_mask_fn=lambda layer_ctx, layer_idx, hi: gen_mask_h2o_with_belong(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            head_idx=hi,
            budget=args.budget,
            seq_len=args.seq_len,
            adaptive_budget=args.adaptive_budget,
        )[0],
    )


def evaluate_one_layer(ctx, args, layer_idx, head_idx, pos_list, model_inputs):
    prefix_patches = build_prefix_patches(
        ctx=ctx,
        args=args,
        target_layer=layer_idx,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )

    artifacts = capture_layer_artifacts(
        ctx=ctx,
        layer_idx=layer_idx,
        pos_list=pos_list,
        model_inputs=model_inputs,
        layer_to_patch=prefix_patches,
    )
    layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)

    route_mask, _belong, count, hh_sumv_idx, hh_sumv_val, hh_sumk_val = gen_mask_h2o_with_belong(
        ctx=layer_ctx,
        layer_idx=layer_idx,
        pos_list=pos_list,
        head_idx=head_idx,
        budget=args.budget,
        seq_len=args.seq_len,
        adaptive_budget=args.adaptive_budget,
        return_hh_sumv=True,
        return_hh_sumk=True,
    )

    qk_logits = get_qk_logits(
        ctx=layer_ctx,
        layer_idx=layer_idx,
        head_idx=head_idx,
        pos_list=pos_list,
        device=ctx.device,
    )

    q_head = layer_ctx.rope_qkv[layer_idx]["q"].to(ctx.device)[0][head_idx][:, pos_list, :].float()

    visible = int(args.seq_len * args.budget)
    if args.adaptive_budget and (layer_idx == 0 or layer_idx == 1):
        visible = args.seq_len
    recent_budget = visible // 2

    alpha_base = torch.softmax(qk_logits + route_mask.to(torch.float32), dim=-1)
    alpha_avgkv = build_avgk_count_refined_alpha(
        qk_logits=qk_logits,
        q_head=q_head,
        mask=route_mask,
        count=count,
        hh_sumk_idx=hh_sumv_idx,
        hh_sumk_val=hh_sumk_val,
        pos_list=pos_list,
        recent_budget=recent_budget,
    )

    v_head = layer_ctx.rope_qkv[layer_idx]["v"].to(ctx.device)[0][head_idx].float()
    v_gt = (
        layer_ctx.attn_output[layer_idx]["output"][0, pos_list]
        .permute(1, 0, 2)
        .to(ctx.device)[head_idx]
        .float()
    )

    v_base = alpha_base.float() @ v_head.float()
    v_avgkv = build_avgv_refined_v(
        alpha=alpha_avgkv,
        v_head=v_head,
        hh_sumv_idx=hh_sumv_idx,
        hh_sumv_val=hh_sumv_val,
        count=count,
    )

    base_metric = v_l2_per_pos_from_v(v_base, v_gt)
    avgkv_metric = v_l2_per_pos_from_v(v_avgkv, v_gt)

    base_mean = float(base_metric.mean().item())
    avgkv_mean = float(avgkv_metric.mean().item())
    improvement = base_mean - avgkv_mean

    return {
        "layer": int(layer_idx),
        "base_v_l2": base_mean,
        "avgkv_v_l2": avgkv_mean,
        "improvement": improvement,
        "improved": bool(improvement > 0.0),
        "base_metric_per_pos": base_metric.detach().cpu(),
        "avgkv_metric_per_pos": avgkv_metric.detach().cpu(),
    }


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    validate_args(
        args,
        num_layers=ctx.model_config.num_hidden_layers,
        num_heads=ctx.model_config.num_attention_heads,
    )

    head_idx = resolve_head_indices(args, ctx.model_config.num_attention_heads)
    pos_end = args.seq_len if args.pos_end is None else min(args.pos_end, args.seq_len)
    pos_list = list(range(args.pos_start, pos_end))
    if len(pos_list) == 0:
        raise ValueError("Empty pos_list after applying --pos-start/--pos-end")

    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers if args.layers is None else False,
        ctx.model_config.num_hidden_layers,
    )

    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    out_dir = resolve_output_dir(args)

    print("===== Check-AvgKV-All-Layers =====")
    print(
        f"layers={layer_idx_list}, heads={head_idx}, budget={args.budget:g}, "
        f"prefix_mode={args.prefix_mode}, strategy={args.strategy}"
    )

    per_layer = []
    for layer_idx in layer_idx_list:
        stats = evaluate_one_layer(
            ctx=ctx,
            args=args,
            layer_idx=layer_idx,
            head_idx=head_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        per_layer.append(stats)
        print(
            f"layer={layer_idx:02d} base={stats['base_v_l2']:.3f} "
            f"avgKV={stats['avgkv_v_l2']:.3f} improvement={stats['improvement']:.3f}"
        )

    improved_cnt = sum(1 for x in per_layer if x["improved"])
    total = len(per_layer)
    mean_improve = float(sum(x["improvement"] for x in per_layer) / max(total, 1))

    print("----- Summary -----")
    print(f"improved_layers={improved_cnt}/{total}")
    print(f"mean_improvement={mean_improve:.3f}")

    tsv_path = os.path.join(out_dir, "layerwise_v_l2.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("layer\tbase_v_l2\tavgkv_v_l2\timprovement\timproved\n")
        for x in per_layer:
            f.write(
                f"{x['layer']}\t{x['base_v_l2']:.3f}\t{x['avgkv_v_l2']:.3f}\t"
                f"{x['improvement']:.3f}\t{int(x['improved'])}\n"
            )

    save_obj = {
        "config": vars(args),
        "layer_idx_list": layer_idx_list,
        "head_idx": head_idx,
        "pos_list": pos_list,
        "per_layer": per_layer,
        "summary": {
            "improved_layers": int(improved_cnt),
            "total_layers": int(total),
            "mean_improvement": mean_improve,
        },
    }
    pt_path = os.path.join(out_dir, "layerwise_v_l2_stats.pt")
    torch.save(save_obj, pt_path)

    print(f"Saved layerwise table to: {tsv_path}")
    print(f"Saved layerwise stats to: {pt_path}")


if __name__ == "__main__":
    main()
