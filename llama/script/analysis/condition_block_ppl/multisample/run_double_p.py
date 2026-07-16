import argparse
import math
from types import SimpleNamespace

from ..runner_cond_block import _full_attention_stats
from ..runner_double_p import _parse_p_setting, run_for_setting
from .common import add_common_args, metric_record, run_multisample


DEFAULT_P_SETTINGS = [
    (0.50, 0.10),
    (0.65, 0.15),
    (0.75, 0.20),
    (0.85, 0.30),
    (0.90, 0.50),
    (0.95, 0.70),
    (1.00, 1.00),
]


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(
            description=(
                "Run Double-P on fixed chunks with dense prefill, sparse "
                "teacher-forced continuation, and PPL over every chunk target."
            )
        )
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Dense prefill length. Defaults to three quarters of seq-len.",
    )
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument(
        "--p-settings",
        type=_parse_p_setting,
        nargs="+",
        default=DEFAULT_P_SETTINGS,
        metavar="P1:P2",
    )
    args = parser.parse_args()
    if args.prompt_len is None:
        args.prompt_len = 3 * args.seq_len // 4
    if not 0 < args.prompt_len < args.seq_len - 1:
        parser.error("--prompt-len must leave at least one scored sparse query")
    if args.cluster_size <= 0 or args.kmeans_iters <= 0:
        parser.error("--cluster-size and --kmeans-iters must be positive")
    if args.sink_tokens < 0 or args.window_size < 0:
        parser.error("--sink-tokens and --window-size must be non-negative")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be non-negative")
    p1_values = [float(p1) for p1, _p2 in args.p_settings]
    if len(p1_values) != len(set(p1_values)):
        parser.error("Every Double-P setting must have a unique p1 value")
    args.p2_by_p1 = {
        float(p1): float(p2) for p1, p2 in args.p_settings
    }
    return args


def _combined_full_chunk_budget(
    *,
    num_heads,
    num_layers,
    prefix_positions,
    tail_budget,
):
    prefix_stats = _full_attention_stats(
        n_heads=num_heads,
        pos_list=prefix_positions,
    )
    prefix_tokens = int(prefix_stats["total_available"]) * int(num_layers)
    tail_stats = tail_budget["aggregate"]
    hybrid_tokens = prefix_tokens + int(tail_stats["hybrid_tokens"])
    total_available = prefix_tokens + int(tail_stats["total_available"])
    return hybrid_tokens / total_available


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    prefix_positions = [position for position in pos_list if position < args.prompt_len]
    tail_positions = [position for position in pos_list if position >= args.prompt_len]
    if not tail_positions:
        raise ValueError("Double-P aligned evaluation has no sparse query positions")
    if pos_list != list(range(len(pos_list))):
        raise ValueError("Double-P aligned evaluation expects contiguous positions from zero")

    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        prompt_len=args.prompt_len,
        cluster_size=args.cluster_size,
        kmeans_iters=args.kmeans_iters,
        sink_tokens=args.sink_tokens,
        window_size=args.window_size,
        full_attention_layers=args.full_attention_layers,
    )
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    results = {}
    for p1 in settings:
        p1 = float(p1)
        p2 = float(args.p2_by_p1[p1])
        if math.isclose(p1, 1.0) and math.isclose(p2, 1.0):
            logits = ref_logits
            measured_budget = 1.0
            tail_measured_budget = 1.0
            layer_patches = {}
        else:
            tail_logits, layer_patches, tail_budget = run_for_setting(
                ctx=ctx,
                args=runner_args,
                p1=p1,
                p2=p2,
                layer_idx_list=layer_idx_list,
                pos_list=tail_positions,
                model_inputs=model_inputs,
            )
            logits = ref_logits.clone()
            logits[:, args.prompt_len :, :] = tail_logits
            tail_measured_budget = float(
                tail_budget["aggregate"]["mean_budget_causal"]
            )
            measured_budget = _combined_full_chunk_budget(
                num_heads=ctx.model_config.num_attention_heads,
                num_layers=ctx.model_config.num_hidden_layers,
                prefix_positions=prefix_positions,
                tail_budget=tail_budget,
            )

        record = metric_record(ref_logits, logits, labels, measured_budget)
        record.update(
            {
                "p1": p1,
                "p2": p2,
                "prompt_len": int(args.prompt_len),
                "num_dense_scored_queries": len(prefix_positions),
                "num_sparse_scored_queries": len(tail_positions),
                "tail_measured_budget": tail_measured_budget,
            }
        )
        results[p1] = record
        print(
            f"[double_p] p1={p1:g}, p2={p2:g}, "
            f"ppl={record['student_ppl']:.6f}, "
            f"full_chunk_budget={measured_budget:.6f}, "
            f"tail_budget={tail_measured_budget:.6f}"
        )
        if logits is not ref_logits:
            del logits, tail_logits, layer_patches
    return results


def main():
    args = parse_args()
    settings = [float(p1) for p1, _p2 in args.p_settings]
    run_multisample(
        args,
        method="double_p",
        settings=settings,
        evaluate_sample=evaluate_sample,
        setting_label="p1",
    )


if __name__ == "__main__":
    main()
