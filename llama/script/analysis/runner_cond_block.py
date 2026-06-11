"""
Run block-condition hybrid attention through all layers and report PPL.

For each layer and eps value, contiguous block clusters are used as the
compressed attention units.  Clusters with condition > eps are expanded to
full attention over their members; the rest keep one averaged KV unit.
"""

import argparse
import copy
import math
import os
import time

import torch
from tqdm import tqdm

from .condition_block import _resolve_block_size
from .online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from .runner import load_context, resolve_layers
from .runner_utils import (
    mean_nll_and_ppl,
    nll_to_ppl,
    set_seed,
    str_to_torch_dtype,
)
from .sanity import compute_metrics, get_tail_labels, move_model_inputs_to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run condition-thresholded block hybrid attention for all layers."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=None,
        help="Evaluation sample start. If unset, use --start.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--budget",
        type=float,
        default=0.1,
        help="Used to derive block_size=round(1/budget) unless --block-size is set.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Contiguous cluster size. Overrides block_size=round(1/budget).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 5.0],
        help="Condition thresholds. Clusters with condition > eps use full attention.",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-layer-patches",
        action="store_true",
        help="Save final per-layer patch tensors for each eps. This can be large.",
    )
    args = parser.parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    if args.layers is None:
        args.all_layers = True
    return args


def _resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        sample_tag = f"{args.dataset}_{args.eval_start}"
        out_dir = os.path.join("..", "result", sample_tag, "condition_block_runner")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _condition_tensor_for_blocks(q, k_head, v_head, pos, block_size):
    total_available = int(pos) + 1
    scale = math.sqrt(q.numel())
    s_vals = []
    deltas = []
    b_cs = []
    b_all = torch.norm(v_head[:total_available].float(), p=2, dim=-1).max()

    for start in range(0, total_available, block_size):
        end = min(start + block_size, total_available)
        k_cluster = k_head[start:end].float()
        v_cluster = v_head[start:end].float()
        k_bar = k_cluster.mean(dim=0)
        qk_cluster = torch.mv(k_cluster, q.float()) / scale
        s_c = torch.dot(q.float(), k_bar.float()) / scale
        deltas.append((qk_cluster - s_c).abs().max())
        b_cs.append(torch.norm(v_cluster, p=2, dim=-1).max())
        s_vals.append(s_c)

    size_vals = []
    for start in range(0, total_available, block_size):
        size_vals.append(float(min(start + block_size, total_available) - start))

    s_tensor = torch.stack(s_vals).float()
    size_tensor = torch.tensor(size_vals, device=s_tensor.device, dtype=torch.float32)
    delta_tensor = torch.stack(deltas).float()
    b_c_tensor = torch.stack(b_cs).float()
    p_tensor = torch.softmax(torch.log(size_tensor) + s_tensor, dim=0)
    denom = (p_tensor * torch.cosh(delta_tensor)).sum().clamp_min(1e-30)
    return p_tensor * (
        2.0 * b_all * (torch.cosh(delta_tensor) - 1.0) / denom
        + 2.0 * b_c_tensor * torch.tanh(delta_tensor / 2.0)
    )


def _hybrid_head_output_for_pos(q, k_head, v_head, pos, block_size, eps):
    total_available = int(pos) + 1
    scale = math.sqrt(q.numel())
    conditions = _condition_tensor_for_blocks(q, k_head, v_head, pos, block_size)

    log_weight_parts = []
    value_parts = []
    selected_clusters = 0
    selected_tokens = 0
    cluster_idx = 0

    for start in range(0, total_available, block_size):
        end = min(start + block_size, total_available)
        k_cluster = k_head[start:end].float()
        v_cluster = v_head[start:end].float()
        if float(conditions[cluster_idx].item()) > eps:
            logits = torch.mv(k_cluster, q.float()) / scale
            log_weight_parts.append(logits)
            value_parts.append(v_cluster)
            selected_clusters += 1
            selected_tokens += end - start
        else:
            k_bar = k_cluster.mean(dim=0)
            v_bar = v_cluster.mean(dim=0, keepdim=True)
            s_c = torch.dot(q.float(), k_bar.float()) / scale
            log_weight_parts.append((math.log(end - start) + s_c).reshape(1))
            value_parts.append(v_bar)
        cluster_idx += 1

    log_weights = torch.cat(log_weight_parts, dim=0)
    values = torch.cat(value_parts, dim=0)
    weights = torch.softmax(log_weights, dim=0)
    out = weights.unsqueeze(-1) * values

    total_clusters = int(conditions.numel())
    hybrid_tokens = selected_tokens + (total_clusters - selected_clusters)
    stats = {
        "clusters": total_clusters,
        "selected_clusters": selected_clusters,
        "selected_tokens": selected_tokens,
        "hybrid_tokens": hybrid_tokens,
        "total_available": total_available,
    }
    return out.sum(dim=0), stats


def build_condition_block_patch(ctx, layer_idx, artifacts, pos_list, block_size, eps):
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    output = artifacts["attn_output"][0, pos_list].permute(1, 0, 2).to(ctx.device).clone()

    n_heads = q_all.shape[0]
    n_pos = len(pos_list)
    budget_stats = {
        "rows": 0,
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": 0,
        "hybrid_tokens": 0,
        "total_available": 0,
    }

    for h in range(n_heads):
        k_head = k_all[h]
        v_head = v_all[h]
        for i, pos in enumerate(pos_list):
            q = q_all[h, pos]
            out, stats = _hybrid_head_output_for_pos(
                q=q,
                k_head=k_head,
                v_head=v_head,
                pos=pos,
                block_size=block_size,
                eps=eps,
            )
            output[h, i] = out.to(output.dtype)
            budget_stats["rows"] += 1
            for key in (
                "clusters",
                "selected_clusters",
                "selected_tokens",
                "hybrid_tokens",
                "total_available",
            ):
                budget_stats[key] += int(stats[key])

    layer = ctx.model.model.layers[layer_idx]
    patch_hidden = layer.self_attn.o_proj(
        output.permute(1, 0, 2).reshape(n_pos, -1).to(ctx.device)
    )
    return patch_hidden.detach(), budget_stats


def _summarize_budget(stats, seq_len):
    rows = max(int(stats["rows"]), 1)
    total_available = max(int(stats["total_available"]), 1)
    budget_causal = float(stats["hybrid_tokens"] / total_available)
    budget_seq_fraction = float(stats["hybrid_tokens"] / (rows * seq_len))
    return {
        **stats,
        "mean_hybrid_tokens": float(stats["hybrid_tokens"] / rows),
        "mean_budget_causal": budget_causal,
        "mean_budget_visible": budget_causal,
        "mean_budget_seq_fraction": budget_seq_fraction,
        "mean_budget_seq": budget_seq_fraction,
        "mean_selected_clusters": float(stats["selected_clusters"] / rows),
        "mean_selected_tokens": float(stats["selected_tokens"] / rows),
        "selected_cluster_ratio": float(
            stats["selected_clusters"] / max(int(stats["clusters"]), 1)
        ),
    }


def _merge_stats(total, add):
    for key, value in add.items():
        total[key] = int(total.get(key, 0)) + int(value)


def run_for_eps(ctx, args, eps, layer_idx_list, pos_list, model_inputs):
    layer_to_patch = {}
    budget_stats = {}
    aggregate_stats = {}

    layer_iter = tqdm(
        layer_idx_list,
        desc=f"eps={eps:g}",
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        t0 = time.time()
        layer_iter.set_postfix(layer=int(layer_idx))
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=pos_list,
            model_inputs=model_inputs,
            layer_to_patch=layer_to_patch,
        )
        layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
        patch_hidden, layer_budget_stats = build_condition_block_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            block_size=args.block_size,
            eps=eps,
        )
        layer_to_patch[layer_idx] = patch_hidden
        budget_stats[int(layer_idx)] = _summarize_budget(
            layer_budget_stats, seq_len=args.seq_len
        )
        _merge_stats(aggregate_stats, layer_budget_stats)
        tqdm.write(
            f"[eps={eps:g}] layer {layer_idx} done in {time.time() - t0:.2f}s; "
            f"mean budget visible="
            f"{budget_stats[int(layer_idx)]['mean_budget_visible']:.6f}"
        )
        del artifacts, layer_ctx
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


def main():
    set_seed(42)
    args = parse_args()
    if args.eval_start != args.start:
        args = copy.copy(args)
        args.start = args.eval_start
    args.block_size = _resolve_block_size(args)
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
    output_dir = _resolve_output_dir(args)

    with torch.no_grad():
        ref_logits = ctx.model(**model_inputs, use_cache=False).logits[:, pos_list, :].float()
    teacher_nll, teacher_ppl = mean_nll_and_ppl(ref_logits, labels)
    print(
        f"[teacher] nll={teacher_nll:.6f}, ppl={teacher_ppl:.6f}; "
        f"block_size={args.block_size}, layers={layer_idx_list}"
    )

    summary = {
        "config": vars(args),
        "block_size": int(args.block_size),
        "layers": layer_idx_list,
        "teacher_nll": teacher_nll,
        "teacher_ppl": teacher_ppl,
        "eps": {},
    }

    for eps in args.eps:
        print(f"\n[condition-block runner] eps={eps:g}")
        student_logits, layer_to_patch, budget = run_for_eps(
            ctx=ctx,
            args=args,
            eps=float(eps),
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        metrics = compute_metrics(ref_logits, student_logits, labels)
        student_ppl = nll_to_ppl(metrics["student_nll"])
        metrics["teacher_ppl"] = teacher_ppl
        metrics["student_ppl"] = student_ppl
        summary["eps"][float(eps)] = {
            "metrics": metrics,
            "budget": budget,
        }
        if args.save_layer_patches:
            patch_path = os.path.join(output_dir, f"eps_{eps:g}_layer_patches.pt")
            torch.save({k: v.detach().cpu() for k, v in layer_to_patch.items()}, patch_path)
            summary["eps"][float(eps)]["patch_path"] = patch_path
        print(
            f"[eps={eps:g}] student_nll={metrics['student_nll']:.6f}, "
            f"student_ppl={student_ppl:.6f}, "
            f"kl={metrics['sanity_kl']:.6f}, "
            f"equiv_budget_causal="
            f"{budget['aggregate']['mean_budget_causal']:.6f}, "
            f"seq_fraction={budget['aggregate']['mean_budget_seq_fraction']:.6f}"
        )
        del student_logits, layer_to_patch
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "runner_cond_block_summary.pt")
    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
