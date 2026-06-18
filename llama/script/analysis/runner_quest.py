"""
Run the QUEST query-aware page selection baseline through selected layers.

Keys and values are grouped into contiguous pages. For each query and page,
QUEST uses the per-dimension minimum and maximum keys to upper-bound the
largest QK score in that page, selects the top-K pages, and performs exact
attention over all visible tokens in those pages. Layers 0 and 1 keep full
attention, following the QUEST paper.
"""

import argparse
import copy
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

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
from .sanity import (
    compute_metrics,
    grouped_query_heads,
    get_tail_labels,
    move_model_inputs_to_device,
)


FULL_ATTENTION_LAYERS = 2


def _model_output_name(model):
    return str(model).rstrip("/").split("/")[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the QUEST query-aware page selection baseline."
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
        "--page-size",
        type=int,
        default=16,
        help="Number of contiguous KV tokens in each QUEST page.",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0],
        help=(
            "Token-budget fractions of seq_len. Each value is rounded up to a "
            "whole number of pages."
        ),
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-layer-patches",
        action="store_true",
        help="Save final per-layer patch tensors for each budget.",
    )
    args = parser.parse_args()

    if args.eval_start is None:
        args.eval_start = args.start
    if args.layers is None:
        args.all_layers = True
    if args.seq_len <= 0:
        parser.error("--seq-len must be > 0")
    if args.page_size <= 0:
        parser.error("--page-size must be > 0")
    if not args.budgets or any(budget <= 0 or budget > 1 for budget in args.budgets):
        parser.error("--budgets values must be in (0, 1]")
    return args


def _resolve_output_dir(args):
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        sample_tag = f"{args.dataset}_{args.eval_start}"
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "result"
            / _model_output_name(args.model)
            / sample_tag
            / "quest_runner"
            / f"page_size={args.page_size}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _resolve_page_budget(seq_len, budget, page_size):
    requested_tokens = max(1, int(seq_len * budget))
    return math.ceil(requested_tokens / page_size), requested_tokens


def _pad_pages(x, page_size):
    n_heads, seq_len = x.shape[:2]
    tail_shape = x.shape[2:]
    n_pages = math.ceil(seq_len / page_size)
    pad_len = n_pages * page_size - seq_len
    if pad_len > 0:
        pad = torch.zeros(
            (n_heads, pad_len, *tail_shape),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)
    return x.reshape(n_heads, n_pages, page_size, *tail_shape), n_pages


def _gather_page_prefix(prefix_tensor, prefix_idx):
    n_heads, n_pages, page_size = prefix_tensor.shape[:3]
    tail_shape = prefix_tensor.shape[3:]
    n_query = prefix_idx.shape[0]
    expanded = prefix_tensor.unsqueeze(1).expand(
        n_heads, n_query, n_pages, page_size, *tail_shape
    )
    gather_idx = prefix_idx.view(
        1, n_query, n_pages, 1, *([1] * len(tail_shape))
    ).expand(n_heads, n_query, n_pages, 1, *tail_shape)
    return torch.gather(expanded, dim=3, index=gather_idx).squeeze(3)


def _build_page_metadata(k_all, v_all, page_size):
    k_page, n_pages = _pad_pages(k_all.float(), page_size)
    v_page, _ = _pad_pages(v_all.float(), page_size)
    device = k_all.device
    seq_len = k_all.shape[1]

    token_idx = torch.arange(n_pages * page_size, device=device).reshape(
        n_pages, page_size
    )
    valid_token = token_idx < seq_len
    valid_k = valid_token.view(1, n_pages, page_size, 1)
    k_for_max = k_page.masked_fill(~valid_k, float("-inf"))
    k_for_min = k_page.masked_fill(~valid_k, float("inf"))

    return {
        "k_page": k_page,
        "v_page": v_page,
        "token_idx": token_idx,
        "valid_token": valid_token,
        "k_prefix_max": k_for_max.cummax(dim=2).values,
        "k_prefix_min": k_for_min.cummin(dim=2).values,
        "page_starts": torch.arange(n_pages, device=device) * page_size,
        "page_valid_counts": valid_token.sum(dim=1),
    }


def _quest_outputs_for_queries(
    *,
    q_pos,
    pos_tensor,
    metadata,
    page_size,
    page_budget,
    share_selection_across_heads=False,
):
    n_heads, n_query, head_dim = q_pos.shape
    n_pages = metadata["page_starts"].numel()
    scale = math.sqrt(head_dim)

    raw_prefix_len = pos_tensor[:, None] - metadata["page_starts"][None, :] + 1
    prefix_len = raw_prefix_len.clamp(min=0, max=page_size)
    size = torch.minimum(
        prefix_len, metadata["page_valid_counts"][None, :]
    ).long()
    page_exists = size > 0
    prefix_idx = (size - 1).clamp_min(0)

    k_max = _gather_page_prefix(metadata["k_prefix_max"], prefix_idx)
    k_min = _gather_page_prefix(metadata["k_prefix_min"], prefix_idx)
    q_for_pages = q_pos[:, :, None, :]
    page_scores = torch.maximum(
        q_for_pages * k_max,
        q_for_pages * k_min,
    ).sum(dim=-1)
    page_scores = page_scores.masked_fill(~page_exists.unsqueeze(0), float("-inf"))

    top_k = min(page_budget, n_pages)
    scores_for_select = (
        page_scores.mean(dim=0, keepdim=True)
        if share_selection_across_heads
        else page_scores
    )
    top_page_idx = torch.topk(scores_for_select, k=top_k, dim=-1).indices
    selected_pages = torch.zeros_like(scores_for_select, dtype=torch.bool)
    selected_pages.scatter_(dim=-1, index=top_page_idx, value=True)
    if share_selection_across_heads:
        selected_pages = selected_pages.expand(n_heads, -1, -1)
    selected_pages = selected_pages & page_exists.unsqueeze(0)

    token_visible = (
        metadata["valid_token"][None, :, :]
        & (metadata["token_idx"][None, :, :] <= pos_tensor[:, None, None])
    )
    token_selected = selected_pages.unsqueeze(-1) & token_visible.unsqueeze(0)
    token_logits = (
        torch.einsum("hqd,hptd->hqpt", q_pos, metadata["k_page"]) / scale
    )
    token_logits = token_logits.masked_fill(~token_selected, float("-inf"))
    weights = torch.softmax(token_logits.flatten(2), dim=-1).reshape_as(token_logits)
    output = torch.einsum("hqpt,hptd->hqd", weights, metadata["v_page"])

    selected_tokens = token_selected.sum()
    total_available = (pos_tensor.long() + 1).sum() * n_heads
    stats = {
        "rows": int(n_heads * n_query),
        "pages": int(page_exists.sum().item() * n_heads),
        "selected_pages": int(selected_pages.sum().item()),
        "selected_tokens": int(selected_tokens.item()),
        "total_available": int(total_available.item()),
    }
    return output, stats


def build_quest_patch(
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    page_size,
    page_budget,
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
    stats = {}
    for kv_head, _out_indices, query_heads in grouped_query_heads(
        list(range(n_heads)),
        ctx.model_config,
        num_kv_heads=k_all.shape[0],
    ):
        k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
        metadata = _build_page_metadata(k_group, v_group, page_size)
        group_output, group_stats = _quest_outputs_for_queries(
            q_pos=q_all[query_heads][:, pos_tensor, :].float(),
            pos_tensor=pos_tensor,
            metadata=metadata,
            page_size=page_size,
            page_budget=page_budget,
            share_selection_across_heads=True,
        )
        output[query_heads] = group_output
        _merge_stats(stats, group_stats)

    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.to(output_dtype)
        .permute(1, 0, 2)
        .reshape(n_pos, -1)
        .to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach(), stats


def _full_attention_stats(n_heads, pos_list):
    total_available = sum(int(pos) + 1 for pos in pos_list) * n_heads
    return {
        "rows": n_heads * len(pos_list),
        "pages": 0,
        "selected_pages": 0,
        "selected_tokens": total_available,
        "total_available": total_available,
    }


def _summarize_budget(stats, seq_len):
    rows = max(int(stats["rows"]), 1)
    total_available = max(int(stats["total_available"]), 1)
    return {
        **stats,
        "mean_selected_tokens": float(stats["selected_tokens"] / rows),
        "mean_budget_causal": float(stats["selected_tokens"] / total_available),
        "mean_budget_visible": float(stats["selected_tokens"] / total_available),
        "mean_budget_seq_fraction": float(
            stats["selected_tokens"] / (rows * seq_len)
        ),
        "mean_budget_seq": float(stats["selected_tokens"] / (rows * seq_len)),
        "mean_selected_pages": float(stats["selected_pages"] / rows),
        "selected_page_ratio": float(
            stats["selected_pages"] / max(int(stats["pages"]), 1)
        ),
    }


def _merge_stats(total, add):
    for key, value in add.items():
        total[key] = int(total.get(key, 0)) + int(value)


def plot_ppl_vs_budget_from_summary(summary_path, out_path=None):
    summary = torch.load(summary_path, map_location="cpu", weights_only=False)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(summary_path), "ppl_vs_budget.png")

    points = []
    for budget, result in summary["budgets"].items():
        measured = result["budget"]["aggregate"]["mean_budget_causal"]
        ppl = result["metrics"]["student_ppl"]
        points.append((float(measured), float(ppl), float(budget)))
    points.sort()

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        marker="o",
        linewidth=1.5,
        color="#2563eb",
        label="QUEST",
    )
    for measured, ppl, requested in points:
        ax.annotate(
            f"{requested:g}",
            (measured, ppl),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    teacher_ppl = summary.get("teacher_ppl")
    if teacher_ppl is not None:
        ax.axhline(
            float(teacher_ppl),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            label="Full attention",
        )
    ax.set_xlabel("Equivalent causal budget")
    ax.set_ylabel("PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.24)
    ax.legend()
    ax.set_title(f"QUEST, page size {summary['page_size']}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run_for_budget(
    ctx,
    args,
    budget,
    page_budget,
    layer_idx_list,
    pos_list,
    model_inputs,
):
    layer_to_patch = {}
    budget_stats = {}
    aggregate_stats = {}

    layer_iter = tqdm(
        layer_idx_list,
        desc=f"budget={budget:g}",
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        t0 = time.time()
        layer_iter.set_postfix(layer=int(layer_idx))
        if layer_idx < FULL_ATTENTION_LAYERS:
            layer_stats = _full_attention_stats(
                n_heads=ctx.model_config.num_attention_heads,
                pos_list=pos_list,
            )
            budget_stats[int(layer_idx)] = _summarize_budget(
                layer_stats, seq_len=args.seq_len
            )
            _merge_stats(aggregate_stats, layer_stats)
            tqdm.write(
                f"[budget={budget:g}] layer {layer_idx} uses full attention; "
                "mean budget visible=1.000000"
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
        patch_hidden, layer_stats = build_quest_patch(
            ctx=layer_ctx,
            layer_idx=layer_idx,
            artifacts=artifacts,
            pos_list=pos_list,
            page_size=args.page_size,
            page_budget=page_budget,
        )
        layer_to_patch[layer_idx] = patch_hidden
        budget_stats[int(layer_idx)] = _summarize_budget(
            layer_stats, seq_len=args.seq_len
        )
        _merge_stats(aggregate_stats, layer_stats)
        tqdm.write(
            f"[budget={budget:g}] layer {layer_idx} done in "
            f"{time.time() - t0:.2f}s; mean budget visible="
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
    layer_idx_list = sorted(
        set(
            resolve_layers(
                args.layers,
                args.all_layers,
                ctx.model_config.num_hidden_layers,
            )
        )
    )
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)
    output_dir = _resolve_output_dir(args)

    with torch.no_grad():
        ref_logits = ctx.model(
            **model_inputs, use_cache=False
        ).logits[:, pos_list, :].float()
    teacher_nll, teacher_ppl = mean_nll_and_ppl(ref_logits, labels)
    print(
        f"[teacher] nll={teacher_nll:.6f}, ppl={teacher_ppl:.6f}; "
        f"page_size={args.page_size}, layers={layer_idx_list}"
    )

    summary = {
        "config": vars(args),
        "page_size": int(args.page_size),
        "layers": layer_idx_list,
        "full_attention_layers": [
            layer_idx
            for layer_idx in layer_idx_list
            if layer_idx < FULL_ATTENTION_LAYERS
        ],
        "teacher_nll": teacher_nll,
        "teacher_ppl": teacher_ppl,
        "budgets": {},
    }

    for budget in args.budgets:
        budget = float(budget)
        page_budget, requested_tokens = _resolve_page_budget(
            args.seq_len, budget, args.page_size
        )
        page_aligned_tokens = page_budget * args.page_size
        print(
            f"\n[QUEST runner] budget={budget:g}, requested_tokens={requested_tokens}, "
            f"top_k_pages={page_budget}, page_aligned_tokens={page_aligned_tokens}"
        )
        student_logits, layer_to_patch, measured_budget = run_for_budget(
            ctx=ctx,
            args=args,
            budget=budget,
            page_budget=page_budget,
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        metrics = compute_metrics(ref_logits, student_logits, labels)
        student_ppl = nll_to_ppl(metrics["student_nll"])
        metrics["teacher_ppl"] = teacher_ppl
        metrics["student_ppl"] = student_ppl
        summary["budgets"][budget] = {
            "requested_tokens": int(requested_tokens),
            "top_k_pages": int(page_budget),
            "page_aligned_tokens": int(page_aligned_tokens),
            "metrics": metrics,
            "budget": measured_budget,
        }

        if args.save_layer_patches:
            patch_path = os.path.join(
                output_dir, f"budget_{budget:g}_layer_patches.pt"
            )
            torch.save(
                {key: value.detach().cpu() for key, value in layer_to_patch.items()},
                patch_path,
            )
            summary["budgets"][budget]["patch_path"] = patch_path

        print(
            f"[budget={budget:g}] student_nll={metrics['student_nll']:.6f}, "
            f"student_ppl={student_ppl:.6f}, kl={metrics['sanity_kl']:.6f}, "
            "equiv_budget_causal="
            f"{measured_budget['aggregate']['mean_budget_causal']:.6f}, "
            "seq_fraction="
            f"{measured_budget['aggregate']['mean_budget_seq_fraction']:.6f}"
        )
        del student_logits, layer_to_patch
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "runner_quest_summary.pt")
    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")
    plot_path = plot_ppl_vs_budget_from_summary(summary_path)
    print(f"Saved PPL vs budget plot to: {plot_path}")


if __name__ == "__main__":
    main()
