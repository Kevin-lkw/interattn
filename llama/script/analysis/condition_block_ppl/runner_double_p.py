"""Decode-faithful Double-P perplexity runner.

Double-P applies sparse attention only after a dense prefill.  This runner
therefore keeps ``[0, prompt_len)`` dense, clusters that fixed prompt, and
computes PPL only on teacher-forced query positions ``[prompt_len, seq_len)``.
It is an eager PyTorch accuracy baseline, not a latency implementation.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ..condition_block_gen.methods.double_p import (
    build_double_p_prompt_clusters,
    double_p_attention,
)
from ..online_routing import capture_layer_artifacts, run_with_multilayer_patches
from ..runner_utils import mean_nll_and_ppl, nll_to_ppl, set_seed, str_to_torch_dtype
from ..runtime import load_context, resolve_layers
from ..sanity import compute_metrics, get_tail_labels, move_model_inputs_to_device
from .double_p_config import (
    DENSE_P_SETTING,
    PAPER_P_SETTING,
    double_p_setting_key,
    parse_p_setting,
)
from .runner_cond_block import (
    _full_attention_stats,
    _merge_stats,
    _model_output_name,
    _summarize_budget,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run decode-faithful Double-P PPL with a fixed dense prompt and "
            "teacher-forced continuation."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Dense prompt length. Defaults to three quarters of seq-len.",
    )
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument(
        "--p-settings",
        type=parse_p_setting,
        nargs="+",
        default=[PAPER_P_SETTING, DENSE_P_SETTING],
        metavar="P1:P2",
        help=(
            "Double-P thresholds. The 1:1 setting is a dense-equivalence check; "
            "the paper's Llama-3.1 setting is 0.95:0.70."
        ),
    )
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--full-attention-layers", type=int, default=2)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None)
    layer_group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-layer-patches", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing runs in the output summary and evaluate only missing settings.",
    )
    args = parser.parse_args()

    if args.seq_len < 2:
        parser.error("--seq-len must be >= 2")
    if args.prompt_len is None:
        args.prompt_len = max(1, 3 * args.seq_len // 4)
    if not 0 < args.prompt_len < args.seq_len:
        parser.error("--prompt-len must satisfy 0 < prompt-len < seq-len")
    if args.cluster_size <= 0:
        parser.error("--cluster-size must be > 0")
    if args.kmeans_iters <= 0:
        parser.error("--kmeans-iters must be > 0")
    if args.sink_tokens < 0 or args.window_size < 0:
        parser.error("--sink-tokens and --window-size must be >= 0")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    if args.layers is None:
        args.all_layers = True
    return args


def _resolve_output_dir(args) -> str:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        sample_tag = f"{args.dataset}_{args.start}"
        output_dir = str(Path(__file__).resolve().parents[3] / "result")
        output_dir = os.path.join(
            output_dir,
            _model_output_name(args.model),
            sample_tag,
            "double_p_ppl",
            (
                f"prompt={args.prompt_len}_cluster={args.cluster_size}_"
                f"iters={args.kmeans_iters}_sink={args.sink_tokens}_"
                f"window={args.window_size}"
            ),
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_double_p_patch(
    *,
    ctx,
    layer_idx,
    artifacts,
    pos_list,
    prompt_len,
    cluster_size,
    kmeans_iters,
    p1,
    p2,
    sink_tokens,
    window_size,
):
    """Build one layer's Double-P attention-output patch."""

    if int(artifacts["q"].shape[0]) != 1:
        raise ValueError("Double-P PPL currently expects batch_size=1")
    q_all = artifacts["q"].to(ctx.device)[0]
    k_all = artifacts["k"].to(ctx.device)[0]
    v_all = artifacts["v"].to(ctx.device)[0]
    n_heads, _seq_len, head_dim = map(int, q_all.shape)
    n_kv_heads = int(k_all.shape[0])
    if n_heads % n_kv_heads != 0:
        raise ValueError("Double-P requires GQA/MHA with query heads divisible by KV heads")

    pos_tensor = torch.tensor(pos_list, device=ctx.device, dtype=torch.long)
    if float(p1) == 1.0 and float(p2) == 1.0:
        # This is the dense reference anchor, not a speed measurement.  Reuse
        # the captured eager result so bf16 reduction-order differences from
        # splitting tokens by cluster cannot move the reference PPL.
        output = (
            artifacts["attn_output"]
            .to(ctx.device)[0, pos_tensor]
            .permute(1, 0, 2)
        )
        stats = _full_attention_stats(n_heads=n_heads, pos_list=pos_list)
    else:
        q_grouped = q_all[:, pos_tensor].reshape(
            n_kv_heads,
            n_heads // n_kv_heads,
            len(pos_list),
            head_dim,
        )
        prompt_clusters = build_double_p_prompt_clusters(
            k_all[:, :prompt_len],
            v_all[:, :prompt_len],
            cluster_size=cluster_size,
            kmeans_iters=kmeans_iters,
            sink_tokens=sink_tokens,
            window_size=window_size,
        )
        output, stats = double_p_attention(
            q_grouped=q_grouped,
            k_all=k_all,
            v_all=v_all,
            prompt_clusters=prompt_clusters,
            pos_tensor=pos_tensor,
            p1=p1,
            p2=p2,
        )
        output = output.reshape(n_heads, len(pos_list), head_dim)

    output_dtype = artifacts["attn_output"].dtype
    layer = ctx.model.model.layers[layer_idx]
    proj_dtype = layer.self_attn.o_proj.weight.dtype
    patch_hidden = layer.self_attn.o_proj(
        output.permute(1, 0, 2)
        .reshape(len(pos_list), -1)
        .to(ctx.device, dtype=proj_dtype)
    )
    return patch_hidden.detach().to(output_dtype), stats


def run_for_setting(ctx, args, p1, p2, layer_idx_list, pos_list, model_inputs):
    layer_to_patch = {}
    budget_stats = {}
    aggregate_stats = {}
    setting = double_p_setting_key(p1, p2)

    if float(p1) == 1.0 and float(p2) == 1.0:
        # Keep the plotted/reference endpoint exactly identical to the teacher.
        # Replaying a bf16 forward through hooks can itself introduce small
        # nondeterministic/reduction-order drift even when injected attention
        # outputs are unchanged.
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
        desc=setting,
        unit="layer",
        dynamic_ncols=True,
    )
    for layer_idx in layer_iter:
        t0 = time.time()
        layer_iter.set_postfix(layer=int(layer_idx))
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
            patch_hidden, layer_stats = build_double_p_patch(
                ctx=ctx,
                layer_idx=layer_idx,
                artifacts=artifacts,
                pos_list=pos_list,
                prompt_len=args.prompt_len,
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
                **model_inputs,
                use_cache=False,
            ).logits[:, pos_list, :].float()
    return logits, layer_to_patch, {
        "aggregate": _summarize_budget(aggregate_stats, seq_len=args.seq_len),
        "by_layer": budget_stats,
    }


def main():
    set_seed(42)
    args = parse_args()
    dtype = str_to_torch_dtype(args.dtype)
    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()

    # q at position i predicts label i+1.  The first query patched here is the
    # first token already appended after prefill, matching Double-P decode.
    pos_list = list(range(args.prompt_len, args.seq_len))
    layer_idx_list = resolve_layers(
        args.layers,
        args.all_layers,
        ctx.model_config.num_hidden_layers,
    )
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    labels = get_tail_labels(ctx, pos_list, ctx.device)
    output_dir = _resolve_output_dir(args)

    with torch.no_grad():
        ref_logits = ctx.model(
            **model_inputs,
            use_cache=False,
        ).logits[:, pos_list, :].float()
    teacher_nll, teacher_ppl = mean_nll_and_ppl(ref_logits, labels)
    print(
        f"[teacher decode-tail] positions={pos_list[0]}..{pos_list[-1]}, "
        f"tokens={len(pos_list)}, nll={teacher_nll:.6f}, ppl={teacher_ppl:.6f}"
    )

    summary_path = os.path.join(output_dir, "runner_double_p_summary.pt")
    if args.resume and os.path.exists(summary_path):
        summary = torch.load(summary_path, map_location="cpu", weights_only=False)
        if summary.get("method") != "double_p":
            raise RuntimeError(f"Not a Double-P summary: {summary_path}")
        previous_teacher_nll = float(summary.get("teacher_nll", float("nan")))
        if not math.isfinite(previous_teacher_nll) or not math.isclose(
            previous_teacher_nll,
            teacher_nll,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Existing summary does not match the current dense reference: "
                f"old_nll={previous_teacher_nll}, new_nll={teacher_nll}."
            )
        summary["config"] = vars(args)
        summary["layers"] = layer_idx_list
        summary["teacher_nll"] = teacher_nll
        summary["teacher_ppl"] = teacher_ppl
        summary.setdefault("runs", {})
        print(
            f"[resume] loaded {len(summary['runs'])} existing settings from "
            f"{summary_path}"
        )
    else:
        summary = {
            "method": "double_p",
            "evaluation": "fixed-dense-prompt_teacher-forced-decode-tail",
            "config": vars(args),
            "layers": layer_idx_list,
            "teacher_nll": teacher_nll,
            "teacher_ppl": teacher_ppl,
            "runs": {},
        }
    for p1, p2 in args.p_settings:
        setting = double_p_setting_key(p1, p2)
        if args.resume and setting in summary["runs"]:
            print(f"[resume] skipping existing setting {setting}")
            continue
        print(f"\n[Double-P] {setting}")
        student_logits, layer_to_patch, budget = run_for_setting(
            ctx=ctx,
            args=args,
            p1=float(p1),
            p2=float(p2),
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        metrics = compute_metrics(ref_logits, student_logits, labels)
        metrics["teacher_ppl"] = teacher_ppl
        metrics["student_ppl"] = nll_to_ppl(metrics["student_nll"])
        record = {
            "p1": float(p1),
            "p2": float(p2),
            "dense_reference": float(p1) == 1.0 and float(p2) == 1.0,
            "metrics": metrics,
            "budget": budget,
        }
        if args.save_layer_patches:
            patch_path = os.path.join(output_dir, f"{setting}_layer_patches.pt")
            torch.save(
                {key: value.detach().cpu() for key, value in layer_to_patch.items()},
                patch_path,
            )
            record["patch_path"] = patch_path
        summary["runs"][setting] = record
        torch.save(summary, summary_path)
        print(
            f"[{setting}] ppl={metrics['student_ppl']:.6f}, "
            f"nll_gap={metrics['nll_gap']:.6g}, kl={metrics['sanity_kl']:.6g}, "
            f"equiv_budget={budget['aggregate']['mean_budget_causal']:.6f}"
        )
        del student_logits, layer_to_patch
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
