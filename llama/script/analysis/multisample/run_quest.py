import argparse
import math
from types import SimpleNamespace

from ..runner_quest import run_for_budget
from .common import add_common_args, metric_record, run_multisample


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(description="Run QUEST on many samples.")
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    return parser.parse_args()


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    results = {}
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        page_size=args.page_size,
    )
    for budget in settings:
        requested_tokens = max(1, int(args.seq_len * budget))
        page_budget = math.ceil(requested_tokens / args.page_size)
        logits, _patches, measured = run_for_budget(
            ctx=ctx,
            args=runner_args,
            budget=float(budget),
            page_budget=page_budget,
            layer_idx_list=layer_idx_list,
            pos_list=pos_list,
            model_inputs=model_inputs,
        )
        measured_budget = measured["aggregate"]["mean_budget_causal"]
        results[float(budget)] = metric_record(
            ref_logits, logits, labels, measured_budget
        )
        print(
            f"[quest] budget={budget:g}, "
            f"ppl={results[float(budget)]['student_ppl']:.6f}, "
            f"measured_budget={measured_budget:.6f}"
        )
        del logits, _patches
    return results


def main():
    args = parse_args()
    run_multisample(
        args,
        method="quest",
        settings=args.budgets,
        evaluate_sample=evaluate_sample,
        setting_label="budget",
    )


if __name__ == "__main__":
    main()
