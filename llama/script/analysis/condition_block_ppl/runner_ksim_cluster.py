"""Run simple K-similarity cluster attention through all layers and report PPL."""

import argparse
import copy
import os
import time

import torch
from tqdm import tqdm

from .condition_ksim_cluster import (
    build_ksim_prefix_tensors,
    ksim_prefix_outputs_for_queries,
)
from ..online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ..runtime import load_context, resolve_layers
from .runner_cond_block import (
    _full_attention_stats,
    _merge_stats,
    _model_output_name,
    _summarize_budget,
    plot_ppl_vs_budget_from_summary,
)
from ..runner_utils import (
    mean_nll_and_ppl,
    nll_to_ppl,
    set_seed,
    str_to_torch_dtype,
)
from ..sanity import (
    compute_metrics,
    get_tail_labels,
    grouped_query_heads,
    move_model_inputs_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple K-similarity cluster attention."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--eval-start", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--cluster-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Average K-sim cluster sizes to evaluate.",
    )
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--full-attention-layers", type=int, default=2)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-layer-patches", action="store_true")
    args = parser.parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    if args.layers is None:
        args.all_layers = True
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    if not args.cluster_sizes or any(size <= 0 for size in args.cluster_sizes):
        parser.error("--cluster-sizes must be positive")
    if args.kmeans_iters <= 0:
        parser.error("--kmeans-iters must be > 0")
    return args


def _resolve_output_dir(args, cluster_size):
    if args.output_dir is not None:
        base_dir = args.output_dir
    else:
        sample_tag = f"{args.dataset}_{args.eval_start}"
        base_dir = os.path.join(
            "..",
            "result",
            _model_output_name(args.model),
            sample_tag,
            "ksim_cluster_runner",
        )
    out_dir = os.path.join(
        base_dir,
        f"cluster={int(cluster_size)}_iters={int(args.kmeans_iters)}",
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_ksim_cluster_patch(ctx, layer_idx, artifacts, pos_list, cluster_size, kmeans_iters):
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
    budget_stats = {}
    for kv_head, _out_indices, query_heads in grouped_query_heads(
        list(range(n_heads)),
        ctx.model_config,
        num_kv_heads=k_all.shape[0],
    ):
        q_pos = q_all[query_heads][:, pos_tensor, :].float()
        prefix = build_ksim_prefix_tensors(
            k_all[kv_head].float(),
            v_all[kv_head].float(),
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
        )
        group_output, group_stats = ksim_prefix_outputs_for_queries(
            q_pos=q_pos,
            pos_tensor=pos_tensor,
            prefix=prefix,
        )
        output[query_heads] = group_output
        _merge_stats(budget_stats, group_stats)

    output = output.to(output_dtype)
    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.permute(1, 0, 2).reshape(n_pos, -1).to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach(), budget_stats


def run_for_cluster_size(ctx, args, cluster_size, layer_idx_list, pos_list, model_inputs):
    layer_to_patch = {}
    budget_stats = {}
    aggregate_stats = {}

    layer_iter = tqdm(
        layer_idx_list,
        desc=f"cluster={cluster_size}",
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        t0 = time.time()
        layer_iter.set_postfix(layer=int(layer_idx))
        if layer_idx < args.full_attention_layers:
            layer_budget_stats = _full_attention_stats(
                n_heads=ctx.model_config.num_attention_heads,
                pos_list=pos_list,
            )
            budget_stats[int(layer_idx)] = _summarize_budget(
                layer_budget_stats,
                seq_len=args.seq_len,
            )
            _merge_stats(aggregate_stats, layer_budget_stats)
            tqdm.write(
                f"[cluster={cluster_size}] layer {layer_idx} uses full attention; "
                f"mean budget visible="
                f"{budget_stats[int(layer_idx)]['mean_budget_visible']:.6f}"
            )
            continue

        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        patch_hidden, layer_budget_stats = build_ksim_cluster_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            cluster_size=cluster_size,
            kmeans_iters=args.kmeans_iters,
        )
        layer_to_patch[layer_idx] = patch_hidden
        budget_stats[int(layer_idx)] = _summarize_budget(
            layer_budget_stats,
            seq_len=args.seq_len,
        )
        _merge_stats(aggregate_stats, layer_budget_stats)
        tqdm.write(
            f"[cluster={cluster_size}] layer {layer_idx} done in {time.time() - t0:.2f}s; "
            f"mean budget visible="
            f"{budget_stats[int(layer_idx)]['mean_budget_visible']:.6f}"
        )
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
    else:
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    return logits, layer_to_patch, {
        "aggregate": _summarize_budget(aggregate_stats, seq_len=args.seq_len),
        "by_layer": budget_stats,
    }


def main():
    set_seed(42)
    args = parse_args()
    if args.eval_start != args.start:
        args = copy.copy(args)
        args.start = args.eval_start
    dtype = str_to_torch_dtype(args.dtype)

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()
    pos_list = list(range(args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)

    with torch.no_grad():
        ref_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    teacher_nll, teacher_ppl = mean_nll_and_ppl(ref_logits, labels)
    print(
        f"[teacher] nll={teacher_nll:.6f}, ppl={teacher_ppl:.6f}; "
        f"cluster_sizes={args.cluster_sizes}, layers={layer_idx_list}"
    )

    for cluster_size in args.cluster_sizes:
        output_dir = _resolve_output_dir(args, cluster_size)
        summary = {
            "config": vars(args),
            "cluster_size": int(cluster_size),
            "kmeans_iters": int(args.kmeans_iters),
            "layers": layer_idx_list,
            "full_attention_layers": [
                layer_idx
                for layer_idx in layer_idx_list
                if layer_idx < args.full_attention_layers
            ],
            "teacher_nll": teacher_nll,
            "teacher_ppl": teacher_ppl,
            "runs": {},
        }
        print(f"\n[ksim cluster runner] cluster_size={cluster_size}")
        student_logits, layer_to_patch, budget = run_for_cluster_size(
            ctx=ctx,
            args=args,
            cluster_size=int(cluster_size),
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        metrics = compute_metrics(ref_logits, student_logits, labels)
        student_ppl = nll_to_ppl(metrics["student_nll"])
        metrics["teacher_ppl"] = teacher_ppl
        metrics["student_ppl"] = student_ppl
        summary["runs"][int(cluster_size)] = {
            "metrics": metrics,
            "budget": budget,
        }
        if args.save_layer_patches:
            patch_path = os.path.join(output_dir, "layer_patches.pt")
            torch.save({k: v.detach().cpu() for k, v in layer_to_patch.items()}, patch_path)
            summary["runs"][int(cluster_size)]["patch_path"] = patch_path
        print(
            f"[cluster={cluster_size}] student_nll={metrics['student_nll']:.6f}, "
            f"student_ppl={student_ppl:.6f}, "
            f"kl={metrics['sanity_kl']:.6f}, "
            f"equiv_budget_causal="
            f"{budget['aggregate']['mean_budget_causal']:.6f}, "
            f"seq_fraction={budget['aggregate']['mean_budget_seq_fraction']:.6f}"
        )
        summary_path = os.path.join(output_dir, "runner_ksim_cluster_summary.pt")
        torch.save(summary, summary_path)
        print(f"Saved summary to: {summary_path}")
        # The existing plotter expects eps-shaped summaries, so keep this runner's
        # plot to downstream comparison scripts for now.
        del student_logits, layer_to_patch
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
