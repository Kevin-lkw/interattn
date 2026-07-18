"""Offline Spearman study against true GQA-group post-Wo block error."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .core import (
    CONDITION_VARIANTS,
    POST_WO_SPECTRAL,
    build_block_prefix,
    cached_layer_head_spectral_norms,
    compute_condition_data,
    exact_block_contribution_errors,
    split_o_proj_weight,
    value_norms_for_variant,
)
from ...online_routing import capture_layer_artifacts
from ...runtime import load_context
from ...runner_utils import set_seed, str_to_torch_dtype
from ...sanity import grouped_query_heads, move_model_inputs_to_device


RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "post_wo_condition_block"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--delta-mode", choices=["exact", "range_bound"], default="range_bound")
    parser.add_argument("--query-start", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_ROOT / "offline_spearman",
    )
    return parser.parse_args()


def _query_positions(args):
    if args.query_start < 0 or args.query_start >= args.seq_len:
        raise ValueError("query-start must be in [0, seq-len)")
    positions = torch.linspace(
        args.query_start, args.seq_len - 1, steps=args.num_queries
    ).round().long().unique(sorted=True)
    return [int(value) for value in positions.tolist()]


def _rankdata(values):
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    i = 0
    while i < values.numel():
        j = i + 1
        while j < values.numel() and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _pearson(x, y):
    if x.numel() < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)
    if float(denom) == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _summarize(rows):
    summary = {"count": len(rows), "variants": {}}
    if not rows:
        return summary
    truth = torch.tensor([row["true_post_error"] for row in rows], dtype=torch.float64)
    for variant in CONDITION_VARIANTS:
        score = torch.tensor([row[variant] for row in rows], dtype=torch.float64)
        valid = torch.isfinite(score) & torch.isfinite(truth) & (truth >= 0)
        xv = score[valid]
        yv = truth[valid]
        positive = (xv > 0) & (yv > 0)
        summary["variants"][variant] = {
            "count": int(valid.sum()),
            "spearman": _pearson(_rankdata(xv), _rankdata(yv)),
            "pearson": _pearson(xv, yv),
            "log10_pearson": _pearson(
                torch.log10(xv[positive]), torch.log10(yv[positive])
            ) if int(positive.sum()) >= 2 else float("nan"),
        }
    cancellation = torch.tensor(
        [row["head_cancellation_ratio"] for row in rows], dtype=torch.float64
    )
    summary["head_cancellation_ratio"] = {
        "mean": float(cancellation.mean()),
        "median": float(cancellation.median()),
        "p10": float(cancellation.quantile(0.1)),
        "p90": float(cancellation.quantile(0.9)),
    }
    return summary


def _true_group_post_error(pre_errors, w_group):
    n_query, n_blocks = pre_errors.shape[1:3]
    post_sum = torch.zeros(
        n_query,
        n_blocks,
        w_group.shape[1],
        device=pre_errors.device,
        dtype=torch.float32,
    )
    head_norm_sum = torch.zeros(
        n_query, n_blocks, device=pre_errors.device, dtype=torch.float32
    )
    for local_head in range(pre_errors.shape[0]):
        contribution = torch.matmul(
            pre_errors[local_head].float(), w_group[local_head].float().transpose(0, 1)
        )
        post_sum.add_(contribution)
        head_norm_sum.add_(torch.linalg.vector_norm(contribution, dim=-1))
        del contribution
    true_error = torch.linalg.vector_norm(post_sum, dim=-1)
    cancellation = true_error / head_norm_sum.clamp_min(1e-30)
    return true_error, cancellation


def run(args):
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions = _query_positions(args)
    ctx = load_context(
        args, dtype=str_to_torch_dtype(args.dtype), device=args.device
    )
    ctx.model.eval()
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    rows = []

    for layer_idx in args.layers:
        print(f"[offline] capture layer={layer_idx}", flush=True)
        artifacts = capture_layer_artifacts(
            ctx=ctx,
            layer_idx=layer_idx,
            pos_list=positions,
            model_inputs=model_inputs,
            layer_to_patch={},
        )
        q_all = artifacts["q"].to(ctx.device)[0].float()
        k_all = artifacts["k"].to(ctx.device)[0].float()
        v_all = artifacts["v"].to(ctx.device)[0].float()
        pos_tensor = torch.tensor(positions, device=ctx.device, dtype=torch.long)
        n_heads = int(q_all.shape[0])
        head_dim = int(q_all.shape[-1])
        layer = ctx.model.model.layers[layer_idx]
        w_all = split_o_proj_weight(
            layer.self_attn.o_proj.weight, n_heads, head_dim
        )
        spectral_all = cached_layer_head_spectral_norms(
            layer.self_attn.o_proj.weight, n_heads, head_dim
        )

        groups = grouped_query_heads(
            list(range(n_heads)), ctx.model_config, num_kv_heads=k_all.shape[0]
        )
        for kv_head, _out_indices, query_heads in groups:
            q_pos = q_all[query_heads][:, pos_tensor]
            k_group = k_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            v_group = v_all[kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            w_group = w_all[query_heads]
            condition_by_variant = {}
            pre_data = None
            pre_prefix = None
            for variant in CONDITION_VARIANTS:
                spectral_group = (
                    spectral_all[query_heads]
                    if variant == POST_WO_SPECTRAL
                    else None
                )
                value_norms = value_norms_for_variant(
                    v_group,
                    w_group,
                    variant,
                    spectral_norms=spectral_group,
                )
                prefix = build_block_prefix(
                    k_group, v_group, value_norms, args.block_size
                )
                data = compute_condition_data(
                    q_pos, pos_tensor, prefix, args.block_size, args.delta_mode
                )
                condition_by_variant[variant] = data["condition"].mean(dim=0)
                if variant == CONDITION_VARIANTS[0]:
                    pre_data = data
                    pre_prefix = prefix

            pre_errors = exact_block_contribution_errors(pre_data, pre_prefix)
            true_error, cancellation = _true_group_post_error(pre_errors, w_group)
            exists = pre_data["cluster_exists"]
            for query_idx, query_pos in enumerate(positions):
                for block_idx in torch.nonzero(
                    exists[query_idx], as_tuple=False
                ).flatten().tolist():
                    size = int(pre_data["size"][query_idx, block_idx])
                    row = {
                        "layer": int(layer_idx),
                        "query_pos": int(query_pos),
                        "kv_head": int(kv_head),
                        "block": int(block_idx),
                        "block_start": int(block_idx * args.block_size),
                        "block_end": int(block_idx * args.block_size + size - 1),
                        "size": size,
                        "true_post_error": float(true_error[query_idx, block_idx]),
                        "head_cancellation_ratio": float(
                            cancellation[query_idx, block_idx]
                        ),
                    }
                    for variant in CONDITION_VARIANTS:
                        row[variant] = float(
                            condition_by_variant[variant][query_idx, block_idx]
                        )
                    rows.append(row)
            del pre_errors, true_error, condition_by_variant, pre_data, pre_prefix
            torch.cuda.empty_cache()
        del artifacts, q_all, k_all, v_all
        torch.cuda.empty_cache()

    fields = [
        "layer",
        "query_pos",
        "kv_head",
        "block",
        "block_start",
        "block_end",
        "size",
        "true_post_error",
        "head_cancellation_ratio",
        *CONDITION_VARIANTS,
    ]
    table_path = args.output_dir / "rows.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "query_positions": positions,
        "overall": _summarize(rows),
        "by_layer": {
            str(layer): _summarize([row for row in rows if row["layer"] == layer])
            for layer in args.layers
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved rows: {table_path}")
    print(f"Saved summary: {summary_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
