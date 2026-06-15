import argparse
from types import SimpleNamespace

from ..condition_block import _resolve_block_size
from ..runner_cond_block import run_for_eps
from .common import add_common_args, metric_record, run_multisample


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(description="Run condition-block on many samples.")
    )
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1, 0.075, 0.05],
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    args = parser.parse_args()
    args.block_size = _resolve_block_size(args)
    return args


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    results = {}
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        block_size=args.block_size,
        delta_mode=args.delta_mode,
        full_attention_layers=args.full_attention_layers,
    )
    for eps in settings:
        logits, _patches, budget = run_for_eps(
            ctx=ctx,
            args=runner_args,
            eps=float(eps),
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        measured_budget = budget["aggregate"]["mean_budget_causal"]
        results[float(eps)] = metric_record(
            ref_logits, logits, labels, measured_budget
        )
        print(
            f"[condition_block] eps={eps:g}, "
            f"ppl={results[float(eps)]['student_ppl']:.6f}, "
            f"budget={measured_budget:.6f}"
        )
        del logits, _patches
    return results


def main():
    args = parse_args()
    run_multisample(
        args,
        method="condition_block",
        settings=args.eps,
        evaluate_sample=evaluate_sample,
        setting_label="eps",
    )


if __name__ == "__main__":
    main()
