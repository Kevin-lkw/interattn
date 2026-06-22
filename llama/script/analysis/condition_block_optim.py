"""
Memory-lean condition-block attention.

This keeps the range-bound condition/routing computation simple, then computes
the hybrid attention output with a block-streaming online softmax.  Compared
with condition_block_single.py, the attention stage does not materialize the
full [heads, queries, blocks, block_size] QK/logit tensor.
"""

import argparse
import contextlib
import os
import time

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from .condition_block import _resolve_block_size
from .condition_block_single import condition_block_single_forward
from .runner import load_context
from .runner_cond_block import (
    _build_block_prefix_tensors,
    _full_attention_stats,
    _gather_prefix,
    _merge_stats,
    _model_output_name,
    _summarize_budget,
)
from .runner_utils import mean_nll_and_ppl, nll_to_ppl, set_seed, str_to_torch_dtype
from .sanity import (
    compute_metrics,
    get_tail_labels,
    grouped_query_heads,
    move_model_inputs_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run memory-lean condition-block hybrid attention."
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
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument(
        "--delta-mode",
        choices=["range_bound"],
        default="range_bound",
        help="Optimized path currently supports the range-bound condition only.",
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--sanity-check",
        choices=["none", "single-forward"],
        default="single-forward",
        help="Compare the first eps against condition_block_single.",
    )
    parser.add_argument("--sanity-tolerance", type=float, default=5e-3)
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
            "condition_block_optim",
        )
    out_dir = os.path.join(base_dir, f"budget={args.budget:g}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _range_bound_selection_and_summaries(*, q_pos, pos_tensor, prefix, block_size, eps):
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = head_dim**0.5

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
    selected = (condition.mean(dim=0, keepdim=True) > eps) & cluster_exists.unsqueeze(0)
    selected = selected.expand(n_heads, -1, -1)
    return selected, z_logits, v_bar, size, cluster_exists


def _online_update(current_m, current_l, current_o, logits, values, active):
    logits = logits.masked_fill(~active, float("-inf"))
    block_m = logits.amax(dim=-1)
    has_any = active.any(dim=-1)
    new_m = torch.maximum(current_m, block_m)
    new_m = torch.where(has_any, new_m, current_m)

    old_scale = torch.exp(current_m - new_m)
    old_scale = torch.where(torch.isfinite(current_m), old_scale, torch.zeros_like(old_scale))
    exp_logits = torch.exp(logits - new_m.unsqueeze(-1)).masked_fill(~active, 0.0)
    block_l = exp_logits.sum(dim=-1)
    new_l = current_l * old_scale + block_l
    block_o = (exp_logits.unsqueeze(-1) * values).sum(dim=-2)
    new_o = current_o * old_scale.unsqueeze(-1) + block_o
    return new_m, new_l, new_o


def _sdpa_full_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **_kwargs,
):
    dropout_p = float(dropout) if module.training else 0.0
    enable_gqa = query.shape[1] != key.shape[1]
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scaling,
        enable_gqa=enable_gqa,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _streaming_hybrid_outputs_for_queries(*, q_pos, pos_tensor, prefix, block_size, eps):
    n_heads, n_query, head_dim = q_pos.shape
    n_blocks = prefix["block_starts"].numel()
    scale = head_dim**0.5
    selected, z_logits, v_bar, size, cluster_exists = _range_bound_selection_and_summaries(
        q_pos=q_pos,
        pos_tensor=pos_tensor,
        prefix=prefix,
        block_size=block_size,
        eps=eps,
    )

    running_m = torch.full(
        (n_heads, n_query), float("-inf"), device=q_pos.device, dtype=torch.float32
    )
    running_l = torch.zeros((n_heads, n_query), device=q_pos.device, dtype=torch.float32)
    running_o = torch.zeros(
        (n_heads, n_query, head_dim), device=q_pos.device, dtype=torch.float32
    )

    for block_idx in range(n_blocks):
        visible = (
            prefix["valid_token"][block_idx].view(1, -1)
            & (prefix["token_idx"][block_idx].view(1, -1) <= pos_tensor.view(-1, 1))
        )
        token_active = selected[:, :, block_idx].unsqueeze(-1) & visible.unsqueeze(0)
        if bool(token_active.any().item()):
            k_block = prefix["k_block"][:, block_idx]
            v_block = prefix["v_block"][:, block_idx]
            token_logits = torch.einsum("hqd,htd->hqt", q_pos, k_block) / scale
            token_values = v_block.unsqueeze(1).expand(n_heads, n_query, -1, -1)
            running_m, running_l, running_o = _online_update(
                running_m,
                running_l,
                running_o,
                token_logits,
                token_values,
                token_active,
            )

        cluster_active = (~selected[:, :, block_idx]) & cluster_exists[:, block_idx].unsqueeze(0)
        if bool(cluster_active.any().item()):
            cluster_logits = z_logits[:, :, block_idx].unsqueeze(-1)
            cluster_values = v_bar[:, :, block_idx].unsqueeze(-2)
            running_m, running_l, running_o = _online_update(
                running_m,
                running_l,
                running_o,
                cluster_logits,
                cluster_values,
                cluster_active.unsqueeze(-1),
            )

    output = running_o / running_l.clamp_min(1e-30).unsqueeze(-1)
    selected_tokens = (selected.long() * size.view(1, n_query, n_blocks)).sum()
    cluster_active_all = (~selected) & cluster_exists.unsqueeze(0)
    stats = {
        "rows": int(n_heads * n_query),
        "clusters": int((cluster_exists.sum() * n_heads).item()),
        "selected_clusters": int(selected.sum().item()),
        "selected_tokens": int(selected_tokens.item()),
        "hybrid_tokens": int((selected_tokens + cluster_active_all.sum()).item()),
        "total_available": int(((pos_tensor.long() + 1).sum() * n_heads).item()),
    }
    return output, stats


class ConditionBlockOptimForward:
    def __init__(
        self,
        *,
        model_config,
        layer_idx_list,
        full_attention_layers,
        block_size,
        eps,
        seq_len,
    ):
        self.model_config = model_config
        self.layer_idx_set = {int(layer_idx) for layer_idx in layer_idx_list}
        self.full_attention_layers = int(full_attention_layers)
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.seq_len = int(seq_len)
        self.stats_by_layer = {}
        self.aggregate_stats = {}

    def should_compress(self, layer_idx):
        return int(layer_idx) in self.layer_idx_set and int(layer_idx) >= self.full_attention_layers

    def record_full_layer(self, layer_idx, n_heads, q_len):
        if layer_idx is None or int(layer_idx) not in self.layer_idx_set:
            return
        layer_idx = int(layer_idx)
        if layer_idx in self.stats_by_layer:
            return
        stats = _full_attention_stats(n_heads=n_heads, pos_list=list(range(int(q_len))))
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
            return _sdpa_full_attention_forward(
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
            raise ValueError("condition_block_optim currently expects batch_size=1.")
        if query.shape[2] != key.shape[2]:
            raise ValueError("condition_block_optim expects use_cache=False.")
        if attention_mask is not None and attention_mask.shape[-1] != key.shape[2]:
            raise ValueError("Unsupported attention_mask shape for condition_block_optim.")

        _batch_size, n_heads, q_len, head_dim = query.shape
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
            group_output, group_stats = _streaming_hybrid_outputs_for_queries(
                q_pos=q_pos,
                pos_tensor=pos_tensor,
                prefix=prefix,
                block_size=self.block_size,
                eps=self.eps,
            )
            output[0, :, query_heads, :] = group_output.permute(1, 0, 2)
            _merge_stats(layer_stats, group_stats)

        self.stats_by_layer[int(layer_idx)] = _summarize_budget(layer_stats, seq_len=self.seq_len)
        _merge_stats(self.aggregate_stats, layer_stats)
        return output.to(query.dtype), None


@contextlib.contextmanager
def condition_block_optim_context(runner):
    original_eager = modeling_llama.eager_attention_forward
    runner.original_eager = original_eager
    modeling_llama.eager_attention_forward = runner.hybrid_attention_forward
    try:
        yield runner
    finally:
        modeling_llama.eager_attention_forward = original_eager
        runner.original_eager = None


def condition_block_optim_forward(
    *,
    ctx,
    model_inputs,
    layer_idx_list,
    full_attention_layers,
    block_size,
    eps,
    seq_len,
):
    runner = ConditionBlockOptimForward(
        model_config=ctx.model_config,
        layer_idx_list=layer_idx_list,
        full_attention_layers=full_attention_layers,
        block_size=block_size,
        eps=eps,
        seq_len=seq_len,
    )
    with condition_block_optim_context(runner):
        with torch.no_grad():
            logits = ctx.model(**model_inputs, use_cache=False).logits.float()
    return logits, runner.summarize()


def run_single_forward_sanity(ctx, model_inputs, layer_idx_list, pos_list, args):
    eps = float(args.eps[0])
    optim_logits, optim_budget = condition_block_optim_forward(
        ctx=ctx,
        model_inputs=model_inputs,
        layer_idx_list=layer_idx_list,
        full_attention_layers=args.full_attention_layers,
        block_size=args.block_size,
        eps=eps,
        seq_len=args.seq_len,
    )
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
    optim_logits = optim_logits[:, pos_list, :].float()
    single_logits = single_logits[:, pos_list, :].float()
    max_abs = float((optim_logits - single_logits).abs().max().item())
    mean_abs = float((optim_logits - single_logits).abs().mean().item())
    print(
        "[sanity single-forward] "
        f"eps={eps:g}, max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
        f"optim_budget={optim_budget['aggregate']['mean_budget_causal']:.6f}, "
        f"single_budget={single_budget['aggregate']['mean_budget_causal']:.6f}"
    )
    if max_abs > args.sanity_tolerance:
        raise RuntimeError(
            "condition_block_optim sanity failed: "
            f"max_abs={max_abs:.6g} > tolerance={args.sanity_tolerance:g}"
        )
    return {
        "eps": eps,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "optim_budget": optim_budget,
        "single_budget": single_budget,
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
    if args.sanity_check == "single-forward":
        summary["sanity"] = run_single_forward_sanity(
            ctx=ctx,
            model_inputs=model_inputs,
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            args=args,
        )

    for eps in args.eps:
        t0 = time.time()
        print(f"\n[condition-block optim] eps={eps:g}")
        student_logits, budget = condition_block_optim_forward(
            ctx=ctx,
            model_inputs=model_inputs,
            layer_idx_list=layer_idx_list,
            full_attention_layers=args.full_attention_layers,
            block_size=args.block_size,
            eps=float(eps),
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
            f"student_ppl={student_ppl:.6f}, kl={metrics['sanity_kl']:.6f}, "
            f"equiv_budget_causal={budget['aggregate']['mean_budget_causal']:.6f}, "
            f"elapsed={summary['eps'][float(eps)]['elapsed_sec']:.2f}s"
        )
        del student_logits
        if torch.cuda.is_available() and str(ctx.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "condition_block_optim_summary.pt")
    torch.save(summary, summary_path)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
