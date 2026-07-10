"""
Run condition-block hybrid attention in a single model forward.

This is the direct-forward counterpart of runner_cond_block.py.  Instead of
capturing each layer and reinjecting per-layer patches through many forwards,
it temporarily replaces the Llama eager attention function.  During the one
student forward, early layers use the original full attention and later layers
compute condition-thresholded contiguous-block hybrid attention directly.
"""

import argparse
import contextlib
import os
import time

import torch
from transformers.models.llama import modeling_llama

from .condition_block import _resolve_block_size
from ..runtime import load_context
from .runner_cond_block import (
    _batched_hybrid_outputs_for_queries,
    _build_block_prefix_tensors,
    _full_attention_stats,
    _merge_stats,
    _model_output_name,
    _summarize_budget,
    run_for_eps,
)
from ..runner_utils import mean_nll_and_ppl, nll_to_ppl, set_seed, str_to_torch_dtype
from ..sanity import (
    compute_metrics,
    get_tail_labels,
    grouped_query_heads,
    move_model_inputs_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run condition-block hybrid attention through one forward pass."
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
        "--budget",
        type=float,
        default=0.1,
        help="Used to derive block_size=round(1/budget) unless --block-size is set.",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5],
        help="Clusters with condition > eps are expanded to token attention.",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument(
        "--full-attention-layers",
        type=int,
        default=2,
        help="Keep the first N layers at original full attention.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sanity-check",
        choices=["none", "full-select", "patch-forward"],
        default="patch-forward",
        help=(
            "full-select runs eps=-inf and checks against full attention; "
            "patch-forward compares single-forward logits against the older "
            "multi-forward patch runner at the first requested eps."
        ),
    )
    parser.add_argument(
        "--sanity-tolerance",
        type=float,
        default=5e-3,
        help="Maximum allowed absolute logit error for --sanity-check full-select.",
    )
    args = parser.parse_args()
    if args.eval_start is None:
        args.eval_start = args.start
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    return args


def _resolve_output_dir(args):
    if args.output_dir is not None:
        base_dir = args.output_dir
    else:
        sample_tag = f"{args.dataset}_{args.eval_start}"
        base_dir = os.path.join(
            "..",
            "result",
            _model_output_name(args.model),
            sample_tag,
            "condition_block_single",
        )
    out_dir = os.path.join(base_dir, f"budget={args.budget:g}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class ConditionBlockSingleForward:
    def __init__(
        self,
        *,
        model_config,
        layer_idx_list,
        full_attention_layers,
        block_size,
        eps,
        delta_mode,
        seq_len,
    ):
        self.model_config = model_config
        self.layer_idx_set = {int(layer_idx) for layer_idx in layer_idx_list}
        self.full_attention_layers = int(full_attention_layers)
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.delta_mode = delta_mode
        self.seq_len = int(seq_len)
        self.stats_by_layer = {}
        self.aggregate_stats = {}

    def should_compress(self, layer_idx):
        return int(layer_idx) in self.layer_idx_set and int(layer_idx) >= self.full_attention_layers

    def should_track(self, layer_idx):
        return layer_idx is not None and int(layer_idx) in self.layer_idx_set

    def record_full_layer(self, layer_idx, n_heads, q_len):
        if not self.should_track(layer_idx):
            return
        layer_idx = int(layer_idx)
        if layer_idx in self.stats_by_layer:
            return
        pos_list = list(range(int(q_len)))
        stats = _full_attention_stats(n_heads=n_heads, pos_list=pos_list)
        self.stats_by_layer[layer_idx] = _summarize_budget(stats, seq_len=self.seq_len)
        _merge_stats(self.aggregate_stats, stats)

    def summarize(self):
        return {
            "aggregate": _summarize_budget(self.aggregate_stats, seq_len=self.seq_len),
            "by_layer": dict(sorted(self.stats_by_layer.items())),
        }

    def hybrid_attention_forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or not self.should_compress(layer_idx):
            self.record_full_layer(layer_idx, query.shape[1], query.shape[2])
            return self.original_eager(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )

        if query.shape[0] != 1:
            raise ValueError("condition_block_single currently expects batch_size=1.")
        if query.shape[2] != key.shape[2]:
            raise ValueError(
                "condition_block_single expects use_cache=False with q_len == kv_len."
            )
        if attention_mask is not None and attention_mask.shape[-1] != key.shape[2]:
            raise ValueError("Unsupported attention_mask shape for condition_block_single.")

        batch_size, n_heads, q_len, head_dim = query.shape
        del batch_size
        pos_tensor = torch.arange(q_len, device=query.device, dtype=torch.long)
        output = torch.empty(
            1,
            q_len,
            n_heads,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        layer_stats = {}

        for kv_head, _out_indices, query_heads in grouped_query_heads(
            list(range(n_heads)),
            self.model_config,
            num_kv_heads=key.shape[1],
        ):
            q_pos = query[0, query_heads].float()
            k_group = key[0, kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            v_group = value[0, kv_head : kv_head + 1].expand(len(query_heads), -1, -1)
            prefix = _build_block_prefix_tensors(k_group, v_group, self.block_size)
            group_output, group_stats = _batched_hybrid_outputs_for_queries(
                q_pos=q_pos,
                pos_tensor=pos_tensor,
                prefix=prefix,
                block_size=self.block_size,
                eps=self.eps,
                delta_mode=self.delta_mode,
                share_selection_across_heads=True,
            )
            output[0, :, query_heads, :] = group_output.permute(1, 0, 2)
            _merge_stats(layer_stats, group_stats)

        self.stats_by_layer[int(layer_idx)] = _summarize_budget(
            layer_stats,
            seq_len=self.seq_len,
        )
        _merge_stats(self.aggregate_stats, layer_stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def condition_block_attention_context(runner):
    original_eager = modeling_llama.eager_attention_forward
    runner.original_eager = original_eager
    modeling_llama.eager_attention_forward = runner.hybrid_attention_forward
    try:
        yield runner
    finally:
        modeling_llama.eager_attention_forward = original_eager
        runner.original_eager = None


def condition_block_single_forward(
    *,
    ctx,
    model_inputs,
    layer_idx_list,
    full_attention_layers,
    block_size,
    eps,
    delta_mode,
    seq_len,
):
    runner = ConditionBlockSingleForward(
        model_config=ctx.model_config,
        layer_idx_list=layer_idx_list,
        full_attention_layers=full_attention_layers,
        block_size=block_size,
        eps=eps,
        delta_mode=delta_mode,
        seq_len=seq_len,
    )
    with condition_block_attention_context(runner):
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits.float()
    return logits, runner.summarize()


def run_full_select_sanity(ctx, model_inputs, ref_logits, layer_idx_list, args):
    sanity_logits, sanity_budget = condition_block_single_forward(
        ctx=ctx,
        model_inputs=model_inputs,
        layer_idx_list=layer_idx_list,
        full_attention_layers=0,
        block_size=args.block_size,
        eps=float("-inf"),
        delta_mode=args.delta_mode,
        seq_len=args.seq_len,
    )
    max_abs = float((sanity_logits - ref_logits).abs().max().item())
    mean_abs = float((sanity_logits - ref_logits).abs().mean().item())
    print(
        "[sanity full-select] "
        f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
        f"equiv_budget={sanity_budget['aggregate']['mean_budget_causal']:.6f}"
    )
    if max_abs > args.sanity_tolerance:
        raise RuntimeError(
            "condition_block_single full-select sanity check failed: "
            f"max_abs={max_abs:.6g} > tolerance={args.sanity_tolerance:g}"
        )
    return {"max_abs": max_abs, "mean_abs": mean_abs, "budget": sanity_budget}


def run_patch_forward_sanity(ctx, model_inputs, layer_idx_list, pos_list, args):
    eps = float(args.eps[0])
    single_logits, single_budget = condition_block_single_forward(
        ctx=ctx,
        model_inputs=model_inputs,
        layer_idx_list=layer_idx_list,
        full_attention_layers=args.full_attention_layers,
        block_size=args.block_size,
        eps=eps,
        delta_mode=args.delta_mode,
        seq_len=args.seq_len,
    )
    patch_logits, _layer_to_patch, patch_budget = run_for_eps(
        ctx=ctx,
        args=args,
        eps=eps,
        layer_idx_list=layer_idx_list,
        pos_list=pos_list,
        model_inputs=model_inputs,
    )
    single_logits = single_logits[:, pos_list, :].float()
    patch_logits = patch_logits.float()
    max_abs = float((single_logits - patch_logits).abs().max().item())
    mean_abs = float((single_logits - patch_logits).abs().mean().item())
    budget_diff = abs(
        float(single_budget["aggregate"]["mean_budget_causal"])
        - float(patch_budget["aggregate"]["mean_budget_causal"])
    )
    print(
        "[sanity patch-forward] "
        f"eps={eps:g}, max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
        f"single_budget={single_budget['aggregate']['mean_budget_causal']:.6f}, "
        f"patch_budget={patch_budget['aggregate']['mean_budget_causal']:.6f}"
    )
    if max_abs > args.sanity_tolerance:
        raise RuntimeError(
            "condition_block_single patch-forward sanity check failed: "
            f"max_abs={max_abs:.6g} > tolerance={args.sanity_tolerance:g}"
        )
    return {
        "eps": eps,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "budget_diff": budget_diff,
        "single_budget": single_budget,
        "patch_budget": patch_budget,
    }


def main():
    set_seed(42)
    args = parse_args()
    args.block_size = _resolve_block_size(args)
    dtype = str_to_torch_dtype(args.dtype)

    if args.eval_start != args.start:
        args.start = args.eval_start

    ctx = load_context(args, dtype=dtype, device=args.device)
    ctx.model.eval()
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    model_inputs = move_model_inputs_to_device(ctx.inputs, ctx.device)
    pos_list = list(range(args.seq_len))
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
        "sanity": None,
        "eps": {},
    }

    if args.sanity_check == "full-select":
        summary["sanity"] = run_full_select_sanity(
            ctx=ctx,
            model_inputs=model_inputs,
            ref_logits=ref_logits,
            layer_idx_list=layer_idx_list,
            args=args,
        )
    elif args.sanity_check == "patch-forward":
        summary["sanity"] = run_patch_forward_sanity(
            ctx=ctx,
            model_inputs=model_inputs,
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            args=args,
        )

    for eps in args.eps:
        t0 = time.time()
        print(f"\n[condition-block single] eps={eps:g}")
        student_logits, budget = condition_block_single_forward(
            ctx=ctx,
            model_inputs=model_inputs,
            layer_idx_list=layer_idx_list,
            full_attention_layers=args.full_attention_layers,
            block_size=args.block_size,
            eps=float(eps),
            delta_mode=args.delta_mode,
            seq_len=args.seq_len,
        )
        student_logits = student_logits[:, pos_list, :]
        metrics = compute_metrics(ref_logits, student_logits, labels)
        student_ppl = nll_to_ppl(metrics["student_nll"])
        metrics["teacher_ppl"] = teacher_ppl
        metrics["student_ppl"] = student_ppl
        summary["eps"][float(eps)] = {
            "metrics": metrics,
            "budget": budget,
            "elapsed_sec": float(time.time() - t0),
        }
        print(
            f"[eps={eps:g}] student_nll={metrics['student_nll']:.6f}, "
            f"student_ppl={student_ppl:.6f}, "
            f"kl={metrics['sanity_kl']:.6f}, "
            f"equiv_budget_causal={budget['aggregate']['mean_budget_causal']:.6f}, "
            f"elapsed={summary['eps'][float(eps)]['elapsed_sec']:.2f}s"
        )
        del student_logits
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "condition_block_single_summary.pt")
    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
