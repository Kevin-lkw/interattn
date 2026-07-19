"""Run an aligned PPL sweep with the additive per-cluster condition score."""

import argparse
from types import SimpleNamespace

from .scores import per_cluster_condition_score
from .. import runner_cond_block as rcb
from ..multisample.common import add_common_args, metric_record, run_multisample


def parse_args():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[
            1e5,
            3e5,
            1e6,
            3e6,
            1e7,
            3e7,
            1e8,
            3e8,
            1e9,
            3e9,
            1e10,
        ],
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument("--method-name", default="per_cluster_condition")
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must be > 0")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    return args


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    results = {}
    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        block_size=args.block_size,
        delta_mode=args.delta_mode,
        full_attention_layers=args.full_attention_layers,
    )
    layers = list(range(ctx.model_config.num_hidden_layers))
    for eps in settings:
        logits, _patches, budget = rcb.run_for_eps(
            ctx=ctx,
            args=runner_args,
            eps=float(eps),
            layer_idx_list=layers,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        measured_budget = budget["aggregate"]["mean_budget_causal"]
        results[float(eps)] = metric_record(
            ref_logits, logits, labels, measured_budget
        )
        print(
            f"[{args.method_name}] eps={eps:g}, "
            f"ppl={results[float(eps)]['student_ppl']:.6f}, "
            f"budget={measured_budget:.6f}"
        )
        del logits, _patches
    return results


def main():
    args = parse_args()
    rcb._condition_score_for_blocks = per_cluster_condition_score
    run_multisample(
        args,
        method=args.method_name,
        settings=args.eps,
        evaluate_sample=evaluate_sample,
        setting_label="eps",
    )


if __name__ == "__main__":
    main()
