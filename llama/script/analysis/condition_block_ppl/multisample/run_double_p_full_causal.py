import argparse
import math
from types import SimpleNamespace

from ..double_p_config import (
    DEFAULT_FULL_CAUSAL_P_SETTINGS,
    p2_by_unique_p1,
    parse_p_setting,
)
from ..runner_double_p_full_causal import run_full_causal_setting
from .common import add_common_args, metric_record, run_multisample


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(
            description=(
                "Run full-causal Double-P on every fixed-chunk query with "
                "periodic causal reclustering and no fixed dense prefill."
            )
        )
    )
    parser.add_argument("--cluster-size", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument(
        "--p-settings",
        type=parse_p_setting,
        nargs="+",
        default=DEFAULT_FULL_CAUSAL_P_SETTINGS,
        metavar="P1:P2",
    )
    args = parser.parse_args()
    if args.cluster_size <= 0 or args.kmeans_iters <= 0:
        parser.error("--cluster-size and --kmeans-iters must be positive")
    if args.sink_tokens < 0 or args.window_size < 0:
        parser.error("--sink-tokens and --window-size must be non-negative")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be non-negative")
    try:
        args.p2_by_p1 = p2_by_unique_p1(args.p_settings)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    if pos_list != list(range(len(pos_list))):
        raise ValueError(
            "Full-causal Double-P expects contiguous query positions from zero"
        )
    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        cluster_size=args.cluster_size,
        kmeans_iters=args.kmeans_iters,
        sink_tokens=args.sink_tokens,
        window_size=args.window_size,
        full_attention_layers=args.full_attention_layers,
    )
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    dense_warmup_queries = min(
        len(pos_list),
        args.sink_tokens + args.window_size + args.cluster_size - 1,
    )
    results = {}
    for p1 in settings:
        p1 = float(p1)
        p2 = float(args.p2_by_p1[p1])
        if math.isclose(p1, 1.0) and math.isclose(p2, 1.0):
            logits = ref_logits
            measured_budget = 1.0
            layer_patches = {}
        else:
            logits, layer_patches, budget = run_full_causal_setting(
                ctx=ctx,
                args=runner_args,
                p1=p1,
                p2=p2,
                layer_idx_list=layer_idx_list,
                pos_list=pos_list,
                model_inputs=model_inputs,
            )
            measured_budget = float(
                budget["aggregate"]["mean_budget_causal"]
            )

        record = metric_record(ref_logits, logits, labels, measured_budget)
        record.update(
            {
                "p1": p1,
                "p2": p2,
                "implementation": "periodic_full_causal_reclustering",
                "fixed_dense_prefill": 0,
                "num_full_causal_queries": len(pos_list),
                "num_natural_dense_warmup_queries": dense_warmup_queries,
                "recluster_stride": int(args.cluster_size),
            }
        )
        results[p1] = record
        print(
            f"[double_p_full_causal] p1={p1:g}, p2={p2:g}, "
            f"ppl={record['student_ppl']:.6f}, "
            f"budget={measured_budget:.6f}"
        )
        if logits is not ref_logits:
            del logits, layer_patches
    return results


def main():
    args = parse_args()
    settings = [float(p1) for p1, _p2 in args.p_settings]
    run_multisample(
        args,
        method="double_p_full_causal",
        settings=settings,
        evaluate_sample=evaluate_sample,
        setting_label="p1",
    )


if __name__ == "__main__":
    main()
