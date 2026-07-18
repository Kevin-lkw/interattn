"""Unified pre/post-Wo PPL sweep with layers 0-1 kept at full attention."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .core import CONDITION_VARIANTS, build_condition_block_patch
from ...online_routing import (
    build_runtime_layer_ctx,
    capture_layer_artifacts,
    run_with_multilayer_patches,
)
from ...sanity import move_model_inputs_to_device
from ..multisample import common


RESULT_ROOT = Path(__file__).resolve().parents[4] / "result" / "post_wo_condition_block"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ppl-protocol",
        choices=[common.LEGACY_PROTOCOL, common.ALIGNED_PROTOCOL],
        default=common.LEGACY_PROTOCOL,
    )
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument(
        "--eps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--variants", nargs="+", choices=CONDITION_VARIANTS, default=list(CONDITION_VARIANTS)
    )
    parser.add_argument(
        "--delta-mode", choices=["exact", "range_bound"], default="range_bound"
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_ROOT / "ppl",
    )
    args = parser.parse_args()
    if args.full_attention_layers != 2:
        parser.error("This unified experiment requires --full-attention-layers 2")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    args.output_root = args.output_dir
    common.validate_common_args(args)
    return args


def _merge_stats(total, add):
    for key, value in add.items():
        total[key] = int(total.get(key, 0)) + int(value)


def _full_attention_stats(n_heads, pos_list):
    total = int(n_heads * sum(int(pos) + 1 for pos in pos_list))
    return {
        "rows": int(n_heads * len(pos_list)),
        "clusters": 0,
        "selected_clusters": 0,
        "selected_tokens": total,
        "hybrid_tokens": total,
        "total_available": total,
    }


def _summarize_budget(stats, seq_len):
    rows = max(int(stats.get("rows", 0)), 1)
    total_available = max(int(stats.get("total_available", 0)), 1)
    return {
        **stats,
        "mean_hybrid_tokens": float(stats.get("hybrid_tokens", 0) / rows),
        "mean_budget_causal": float(
            stats.get("hybrid_tokens", 0) / total_available
        ),
        "mean_budget_seq_fraction": float(
            stats.get("hybrid_tokens", 0) / (rows * seq_len)
        ),
        "mean_selected_clusters": float(
            stats.get("selected_clusters", 0) / rows
        ),
        "mean_selected_tokens": float(stats.get("selected_tokens", 0) / rows),
    }


def run_for_eps(ctx, args, variant, eps, pos_list, model_inputs):
    patches = {}
    aggregate_stats = {}
    by_layer = {}
    layer_iter = tqdm(
        range(ctx.model_config.num_hidden_layers),
        desc=f"{variant} eps={eps:g}",
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        started = time.time()
        if layer_idx < args.full_attention_layers:
            layer_stats = _full_attention_stats(
                int(ctx.model_config.num_attention_heads), pos_list
            )
        else:
            artifacts = capture_layer_artifacts(
                ctx=ctx,
                layer_idx=layer_idx,
                pos_list=pos_list,
                model_inputs=model_inputs,
                layer_to_patch=patches,
            )
            layer_ctx = build_runtime_layer_ctx(ctx, layer_idx, artifacts)
            patch, layer_stats = build_condition_block_patch(
                ctx=layer_ctx,
                layer_idx=layer_idx,
                artifacts=artifacts,
                pos_list=pos_list,
                block_size=args.block_size,
                eps=float(eps),
                delta_mode=args.delta_mode,
                variant=variant,
            )
            patches[layer_idx] = patch
            del artifacts, layer_ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        by_layer[layer_idx] = _summarize_budget(layer_stats, args.seq_len)
        _merge_stats(aggregate_stats, layer_stats)
        layer_iter.set_postfix(
            layer=layer_idx,
            budget=f"{by_layer[layer_idx]['mean_budget_causal']:.3f}",
            sec=f"{time.time() - started:.1f}",
        )

    if patches:
        logits = run_with_multilayer_patches(
            ctx=ctx,
            layer_to_patch=patches,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
    else:
        with torch.no_grad():
            logits = ctx.model(
                **model_inputs, use_cache=False
            ).logits[:, pos_list, :].float()
    return logits, {
        "aggregate": _summarize_budget(aggregate_stats, args.seq_len),
        "by_layer": by_layer,
    }


def _save(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(summary, tmp)
    os.replace(tmp, path)


def _aggregate(summary):
    aggregate = {"teacher": {}, "variants": {}}
    samples = list(summary["samples"].values())
    if samples:
        teacher_nll = torch.tensor(
            [sample["teacher_nll"] for sample in samples], dtype=torch.float64
        )
        tokens = torch.tensor(
            [sample["num_tokens"] for sample in samples], dtype=torch.float64
        )
        corpus_nll = float((teacher_nll * tokens).sum() / tokens.sum())
        aggregate["teacher"] = {
            "samples": len(samples),
            "corpus_nll": corpus_nll,
            "corpus_ppl": math.exp(corpus_nll),
            "mean_sample_ppl": float(
                torch.tensor([sample["teacher_ppl"] for sample in samples]).mean()
            ),
        }
    for variant in summary["config"]["variants"]:
        aggregate["variants"][variant] = {}
        for eps in summary["config"]["eps"]:
            records = []
            for sample in samples:
                record = sample.get("results", {}).get(variant, {}).get(float(eps))
                if record is not None:
                    records.append(record)
            if not records:
                continue
            nll = torch.tensor(
                [record["student_nll"] for record in records], dtype=torch.float64
            )
            counts = torch.tensor(
                [record["num_tokens"] for record in records], dtype=torch.float64
            )
            corpus_nll = float((nll * counts).sum() / counts.sum())
            aggregate["variants"][variant][float(eps)] = {
                "samples": len(records),
                "corpus_nll": corpus_nll,
                "corpus_ppl": math.exp(corpus_nll),
                "mean_sample_ppl": float(
                    torch.tensor([record["student_ppl"] for record in records]).mean()
                ),
                "mean_kl": float(
                    torch.tensor([record["sanity_kl"] for record in records]).mean()
                ),
                "mean_measured_budget": float(
                    torch.tensor(
                        [record["measured_budget"] for record in records]
                    ).mean()
                ),
            }
    return aggregate


def _plot(summary, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for variant, settings in summary["aggregate"]["variants"].items():
        points = sorted(
            (
                record["mean_measured_budget"],
                record["corpus_ppl"],
                float(eps),
            )
            for eps, record in settings.items()
        )
        if not points:
            continue
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=1.5,
            label=variant,
        )
        for budget, ppl, eps in points:
            ax.annotate(
                f"{eps:g}", (budget, ppl), xytext=(4, 4),
                textcoords="offset points", fontsize=7
            )
    teacher = summary["aggregate"].get("teacher", {})
    if teacher:
        ax.axhline(
            teacher["corpus_ppl"], color="#666", linestyle="--", linewidth=1,
            label="full attention"
        )
    ax.set_xlabel("Measured equivalent causal budget")
    ax.set_ylabel("Corpus PPL")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Condition block: pre-Wo vs post-Wo")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    common.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.pt"
    if summary_path.exists():
        summary = torch.load(summary_path, map_location="cpu", weights_only=False)
        current = {
            key: getattr(args, key)
            for key in (
                "model", "dataset", "seq_len", "num_samples", "start_offset",
                "sample_stride", "ppl_protocol", "block_size", "eps", "variants",
                "delta_mode", "full_attention_layers"
            )
        }
        previous = {key: summary["config"].get(key) for key in current}
        if previous != current:
            raise ValueError(
                f"Existing summary config differs. Existing={previous}, current={current}"
            )
    else:
        summary = {
            "config": {
                **vars(args),
                "output_dir": str(args.output_dir),
                "output_root": str(args.output_root),
            },
            "starts": common.sample_starts(args),
            "samples": {},
            "aggregate": {},
        }

    model, tokenizer, encoded, dtype, starts = common.load_model_and_tokens(args)
    for sample_idx, start in enumerate(starts):
        sample = summary["samples"].setdefault(sample_idx, {})
        ctx = common.build_sample_context(
            model=model,
            tokenizer=tokenizer,
            encoded=encoded,
            dtype=dtype,
            device=args.device,
            start=start,
            seq_len=args.seq_len,
            ppl_protocol=args.ppl_protocol,
        )
        prepared = common.prepare_sample(ctx, args.seq_len)
        pos_list, model_inputs, labels, ref_logits, teacher_nll, teacher_ppl = prepared
        sample.update(
            {
                "start": int(start),
                "teacher_nll": float(teacher_nll),
                "teacher_ppl": float(teacher_ppl),
                "num_tokens": int(labels.numel()),
            }
        )
        sample.setdefault("results", {})
        print(
            f"[sample {sample_idx}] start={start}, teacher_ppl={teacher_ppl:.6f}",
            flush=True,
        )
        for variant in args.variants:
            variant_results = sample["results"].setdefault(variant, {})
            for eps in args.eps:
                if float(eps) in variant_results:
                    print(f"[skip] {variant} eps={eps:g}", flush=True)
                    continue
                logits, budget = run_for_eps(
                    ctx, args, variant, float(eps), pos_list, model_inputs
                )
                record = common.metric_record(
                    ref_logits,
                    logits,
                    labels,
                    budget["aggregate"]["mean_budget_causal"],
                )
                record["budget"] = budget
                variant_results[float(eps)] = record
                summary["aggregate"] = _aggregate(summary)
                _save(summary, summary_path)
                _plot(summary, args.output_dir / "ppl_vs_budget.png")
                print(
                    f"[result] {variant} eps={eps:g} "
                    f"ppl={record['student_ppl']:.6f} "
                    f"kl={record['sanity_kl']:.6g} "
                    f"budget={record['measured_budget']:.6f}",
                    flush=True,
                )
                del logits
                torch.cuda.empty_cache()
        del ctx, ref_logits
        torch.cuda.empty_cache()

    summary["aggregate"] = _aggregate(summary)
    _save(summary, summary_path)
    aggregate_path = args.output_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(summary["aggregate"], indent=2) + "\n")
    _plot(summary, args.output_dir / "ppl_vs_budget.png")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"Saved summary: {summary_path}")
    print(f"Saved aggregate: {aggregate_path}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
