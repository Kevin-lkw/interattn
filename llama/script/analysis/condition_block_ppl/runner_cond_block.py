"""
Run block-condition hybrid attention through all layers and report PPL.

Layers 0 and 1 keep full attention. For each later layer and eps value,
contiguous block clusters are used as the compressed attention units.
Clusters with condition > eps are expanded to full attention over their
members; the rest keep one averaged KV unit.
"""

import argparse
import copy
import math
import os
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .condition_block import _resolve_block_size
from ..online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ..runtime import load_context, resolve_layers
from ..runner_utils import (
    mean_nll_and_ppl,
    nll_to_ppl,
    set_seed,
    str_to_torch_dtype,
)
from ..sanity import (
    compute_metrics,
    grouped_query_heads,
    get_tail_labels,
    move_model_inputs_to_device,
)


def _model_output_name(model):
    return str(model).rstrip("/").split("/")[-1]


def parse_args(
    description="Run condition-thresholded block hybrid attention for all layers.",
):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
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
        default=[ 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,0.5],
        help="Condition thresholds. Clusters with condition > eps use full attention.",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
        help=(
            "How to compute delta_C. exact uses per-token QK scores; "
            "range_bound uses per-dimension K min/max upper bounds."
        ),
    )
    parser.add_argument(
        "--full-attention-layers",
        type=int,
        default=2,
        help="Keep the first N layers at full attention. Use 0 to compress every layer.",
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
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    return args


def _resolve_output_dir(args, method_name="condition_block_runner"):
    if args.output_dir is not None:
        base_dir = args.output_dir
    else:
        sample_tag = f"{args.dataset}_{args.eval_start}"
        base_dir = os.path.join(
            "..",
            "result",
            _model_output_name(args.model),
            sample_tag,
            method_name,
        )
    out_dir = os.path.join(base_dir, f"budget={args.budget:g}")
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


def _pad_blocks(x, block_size):
    n_heads, seq_len = x.shape[:2]
    tail_shape = x.shape[2:]
    n_blocks = math.ceil(seq_len / block_size)
    pad_len = n_blocks * block_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *tail_shape),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_blocks, block_size, *tail_shape), n_blocks


def _gather_prefix(prefix_tensor, prefix_idx):
    n_heads, n_blocks, block_size = prefix_tensor.shape[:3]
    tail_shape = prefix_tensor.shape[3:]
    n_query = prefix_idx.shape[0]
    expanded = prefix_tensor.unsqueeze(1).expand(
        n_heads, n_query, n_blocks, block_size, *tail_shape
    )
    gather_idx = prefix_idx.view(1, n_query, n_blocks, 1, *([1] * len(tail_shape))).expand(
        n_heads, n_query, n_blocks, 1, *tail_shape
    )
    return torch.gather(expanded, dim=3, index=gather_idx).squeeze(3)


def _build_block_prefix_tensors(k_all, v_all, block_size):
    k_block, n_blocks = _pad_blocks(k_all.float(), block_size)
    v_block, _ = _pad_blocks(v_all.float(), block_size)
    device = k_all.device
    seq_len = k_all.shape[1]

    token_idx = torch.arange(n_blocks * block_size, device=device).reshape(
        n_blocks, block_size
    )
    valid_token = token_idx < seq_len

    valid_k = valid_token.view(1, n_blocks, block_size, 1)
    k_for_max = k_block.masked_fill(~valid_k, float("-inf"))
    k_for_min = k_block.masked_fill(~valid_k, float("inf"))
    v_norm = torch.norm(v_block, p=2, dim=-1)
    v_norm = v_norm.masked_fill(~valid_token.view(1, n_blocks, block_size), float("-inf"))

    return {
        "k_block": k_block,
        "v_block": v_block,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_cumsum": k_block.cumsum(dim=2),
        "v_cumsum": v_block.cumsum(dim=2),
        "k_prefix_max": k_for_max.cummax(dim=2).values,
        "k_prefix_min": k_for_min.cummin(dim=2).values,
        "v_norm_prefix_max": v_norm.cummax(dim=2).values,
        "block_starts": torch.arange(n_blocks, device=device) * block_size,
        "block_valid_counts": valid_token.sum(dim=1),
    }


def _batched_hybrid_outputs_for_queries(
    *,
    q_pos,
    pos_tensor,
    prefix,
    block_size,
    eps,
    delta_mode,
    share_selection_across_heads=False,
    force_first_last_blocks=False,
):
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = math.sqrt(head_dim)

    raw_prefix_len = pos_tensor[:, None] - prefix["block_starts"][None, :] + 1
    prefix_len = raw_prefix_len.clamp(min=0, max=block_size)
    size = torch.minimum(prefix_len, prefix["block_valid_counts"][None, :]).long()
    cluster_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)
    size_float = size.clamp_min(1).float()

    k_sum = _gather_prefix(prefix["k_cumsum"], prefix_idx)
    v_sum = _gather_prefix(prefix["v_cumsum"], prefix_idx)
    k_bar = k_sum / size_float.view(1, n_query, n_blocks, 1)
    v_bar = v_sum / size_float.view(1, n_query, n_blocks, 1)

    s_c = (q_pos[:, :, None, :] * k_bar).sum(dim=-1) / scale
    block_logits = torch.einsum("hqd,hbtd->hqbt", q_pos, prefix["k_block"]) / scale
    token_visible = (
        prefix["valid_token"][None, :, :]
        & (prefix["token_idx"][None, :, :] <= pos_tensor[:, None, None])
    )

    if delta_mode == "exact":
        centered = (block_logits - s_c.unsqueeze(-1)).abs()
        delta = centered.masked_fill(~token_visible.unsqueeze(0), float("-inf")).amax(dim=-1)
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    elif delta_mode == "range_bound":
        k_max = _gather_prefix(prefix["k_prefix_max"], prefix_idx)
        k_min = _gather_prefix(prefix["k_prefix_min"], prefix_idx)
        q_for_bounds = q_pos[:, :, None, :]
        upper_score = torch.maximum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        lower_score = torch.minimum(q_for_bounds * k_max, q_for_bounds * k_min).sum(
            dim=-1
        ) / scale
        delta = torch.maximum((upper_score - s_c).abs(), (lower_score - s_c).abs())
        delta = delta.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    b_c = _gather_prefix(prefix["v_norm_prefix_max"].unsqueeze(-1), prefix_idx).squeeze(-1)
    b_c = b_c.masked_fill(~cluster_exists.unsqueeze(0), 0.0)
    b_all = b_c.amax(dim=-1)

    z_logits = torch.log(size_float).view(1, n_query, n_blocks) + s_c
    z_logits = z_logits.masked_fill(~cluster_exists.unsqueeze(0), float("-inf"))
    p_tensor = torch.softmax(z_logits, dim=-1)
    denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
    condition = p_tensor * (
        2.0 * b_all.unsqueeze(-1) * (torch.cosh(delta) - 1.0) / denom.unsqueeze(-1)
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    if share_selection_across_heads:
        selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
        selected = selected.expand(n_heads, -1, -1)
    else:
        selected = (condition > eps) & cluster_exists.unsqueeze(0)

    forced = torch.zeros_like(selected)
    newly_forced = torch.zeros_like(selected)
    if force_first_last_blocks:
        block_idx = torch.arange(n_blocks, device=pos_tensor.device)
        last_visible_block = cluster_exists.long().sum(dim=-1) - 1
        forced_by_query = cluster_exists & (
            (block_idx.unsqueeze(0) == 0)
            | (block_idx.unsqueeze(0) == last_visible_block.unsqueeze(1))
        )
        forced = forced_by_query.unsqueeze(0).expand(n_heads, -1, -1)
        newly_forced = forced & ~selected
        selected = selected | forced

    token_selected = selected.unsqueeze(-1) & token_visible.unsqueeze(0)
    token_logits = block_logits.masked_fill(~token_selected, float("-inf"))
    cluster_logits = z_logits.masked_fill(selected | ~cluster_exists.unsqueeze(0), float("-inf"))

    token_max = token_logits.flatten(2).amax(dim=-1)
    cluster_max = cluster_logits.amax(dim=-1)
    max_logit = torch.maximum(token_max, cluster_max).clamp_min(-1e30)

    token_exp = torch.exp(token_logits - max_logit[:, :, None, None]).masked_fill(
        ~token_selected, 0.0
    )
    cluster_active = (~selected) & cluster_exists.unsqueeze(0)
    cluster_exp = torch.exp(cluster_logits - max_logit[:, :, None]).masked_fill(
        ~cluster_active, 0.0
    )
    normalizer = (
        token_exp.sum(dim=(2, 3)) + cluster_exp.sum(dim=2)
    ).clamp_min(1e-30)

    token_num = torch.einsum("hqbt,hbtd->hqd", token_exp, prefix["v_block"])
    cluster_num = (cluster_exp.unsqueeze(-1) * v_bar).sum(dim=2)
    output = (token_num + cluster_num) / normalizer.unsqueeze(-1)

    selected_tokens = (selected.long() * size.view(1, n_query, n_blocks)).sum()
    selected_clusters = selected.sum()
    clusters = cluster_exists.sum() * n_heads
    hybrid_tokens = selected_tokens + (cluster_active.sum())
    total_available = (pos_tensor.long() + 1).sum() * n_heads
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int(clusters.item()),
        "selected_clusters": int(selected_clusters.item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int(hybrid_tokens.item()),
        "total_available": int(total_available.item()),
    }
    if force_first_last_blocks:
        stats.update(
            {
                "forced_clusters": int(forced.sum().item()),
                "forced_tokens": int(
                    (forced.long() * size.view(1, n_query, n_blocks)).sum().item()
                ),
                "newly_forced_clusters": int(newly_forced.sum().item()),
                "newly_forced_tokens": int(
                    (
                        newly_forced.long()
                        * size.view(1, n_query, n_blocks)
                    ).sum().item()
                ),
            }
        )
    return output, stats


def build_condition_block_patch(
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    block_size,
    eps,
    delta_mode="range_bound",
    force_first_last_blocks=False,
):
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
        k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        prefix = _build_block_prefix_tensors(k_group, v_group, block_size)
        group_output, group_stats = _batched_hybrid_outputs_for_queries(
            q_pos=q_pos,
            pos_tensor=pos_tensor,
            prefix=prefix,
            block_size=block_size,
            eps=eps,
            delta_mode=delta_mode,
            share_selection_across_heads=True,
            force_first_last_blocks=force_first_last_blocks,
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


def _full_attention_stats(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * n_heads
    return {
        "rows": n_heads * len(pos_list),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total_available,
        "hybrid_tokens": total_available,
        "total_available": total_available,
    }


def plot_ppl_vs_budget_from_summary(summary_path, out_path=None):
    summary = torch.load(summary_path, map_location="cpu")
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(summary_path), "ppl_vs_equiv_budget_causal.png"
        )

    points = []
    for eps, eps_summary in summary.get("eps", {}).items():
        budget = eps_summary["budget"]["aggregate"]["mean_budget_causal"]
        ppl = eps_summary["metrics"]["student_ppl"]
        points.append((float(budget), float(ppl), float(eps)))

    if not points:
        raise ValueError(f"No eps results found in summary: {summary_path}")

    points.sort(key=lambda item: item[0])
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=1.4, color="#2563eb")
    for x, y, eps in points:
        ax.annotate(
            f"eps={eps:g}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    teacher_ppl = summary.get("teacher_ppl")
    if teacher_ppl is not None:
        ax.axhline(float(teacher_ppl), color="#6b7280", linestyle="--", linewidth=1.0)
        ax.text(xs[0], float(teacher_ppl), "teacher", va="bottom", fontsize=8)

    ax.set_xlabel("equiv_budget_causal")
    ax.set_ylabel("PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.24)
    ax.set_title("PPL vs equiv budget causal")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


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
        if layer_idx < args.full_attention_layers:
            layer_budget_stats = _full_attention_stats(
                n_heads=ctx.model_config.num_attention_heads,
                pos_list=pos_list,
            )
            budget_stats[int(layer_idx)] = _summarize_budget(
                layer_budget_stats, seq_len=args.seq_len
            )
            _merge_stats(aggregate_stats, layer_budget_stats)
            tqdm.write(
                f"[eps={eps:g}] layer {layer_idx} uses full attention; "
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
        patch_hidden, layer_budget_stats = build_condition_block_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            block_size=args.block_size,
            eps=eps,
            delta_mode=args.delta_mode,
            force_first_last_blocks=getattr(
                args, "force_first_last_blocks", False
            ),
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

    if layer_to_patch:
        logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=layer_to_patch,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
    else:
        with torch.no_grad():
            logits = ctx.model(
                **model_inputs, use_cache=False
            ).logits[:, pos_list, :].float()
    return logits, layer_to_patch, {
        "aggregate": _summarize_budget(aggregate_stats, seq_len=args.seq_len),
        "by_layer": budget_stats,
    }


def main(
    args=None,
    *,
    output_method="condition_block_runner",
    runner_label="condition-block runner",
    summary_filename="runner_cond_block_summary.pt",
):
    set_seed(42)
    if args is None:
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
    output_dir = _resolve_output_dir(args, method_name=output_method)

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
        "full_attention_layers": [
            layer_idx
            for layer_idx in layer_idx_list
            if layer_idx < args.full_attention_layers
        ],
        "teacher_nll": teacher_nll,
        "teacher_ppl": teacher_ppl,
        "eps": {},
    }

    for eps in args.eps:
        print(f"\n[{runner_label}] eps={eps:g}")
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

    summary_path = os.path.join(output_dir, summary_filename)
    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")
    plot_path = plot_ppl_vs_budget_from_summary(summary_path)
    print(f"Saved PPL vs budget plot to: {plot_path}")


if __name__ == "__main__":
    main()
