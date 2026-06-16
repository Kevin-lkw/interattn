import argparse

from .common import (
    add_common_args,
    metric_record,
    run_attention_topk_method,
    run_multisample,
)


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(description="Run oracle attention top-k on many samples.")
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    return parser.parse_args()


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    results = {}
    for budget in settings:
        if float(budget) == 1.0:
            logits = ref_logits
            measured_budget = 1.0
        else:
            logits, measured_budget = run_attention_topk_method(
                ctx=ctx,
                budget=budget,
                full_attention_layers=args.full_attention_layers,
                seq_len=args.seq_len,
                pos_list=pos_list,
                model_inputs=model_inputs,
            )
        results[float(budget)] = metric_record(
            ref_logits, logits, labels, measured_budget
        )
        print(
            f"[attention_topk] budget={budget:g}, "
            f"ppl={results[float(budget)]['student_ppl']:.6f}"
        )
        if logits is not ref_logits:
            del logits
    return results


def main():
    args = parse_args()
    run_multisample(
        args,
        method="attention_topk",
        settings=args.budgets,
        evaluate_sample=evaluate_sample,
        setting_label="budget",
    )


if __name__ == "__main__":
    main()
