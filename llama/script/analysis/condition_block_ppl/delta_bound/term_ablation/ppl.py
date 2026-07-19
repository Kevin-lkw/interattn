"""Run aligned PPL sweeps for term1/term2 condition-score variants."""

import argparse
from types import SimpleNamespace

import torch

from ... import runner_cond_block as rcb
from ...multisample.common import add_common_args, metric_record, run_multisample


SCORE_KINDS = ("original", "mass_exp")


def condition_score(
    *,
    p_tensor,
    z_logits,
    delta,
    b_c,
    b_all,
    cluster_exists,
    score_kind,
    term2_weight,
    term1_weight=1.0,
):
    """Return a selection score while preserving the runner's tensor layout."""
    if score_kind == "original":
        denom = (p_tensor * torch.cosh(delta)).sum(dim=-1).clamp_min(1e-30)
        if float(term1_weight) == 1.0 and float(term2_weight) == 1.0:
            return p_tensor * (
                2.0
                * b_all.unsqueeze(-1)
                * (torch.cosh(delta) - 1.0)
                / denom.unsqueeze(-1)
                + 2.0 * b_c * torch.tanh(delta / 2.0)
            )
        term1 = (
            p_tensor
            * 2.0
            * b_all.unsqueeze(-1)
            * (torch.cosh(delta) - 1.0)
            / denom.unsqueeze(-1)
        )
    elif score_kind == "mass_exp":
        mass = torch.softmax(z_logits + delta, dim=-1)
        term1 = 2.0 * b_all.unsqueeze(-1) * mass
    else:
        raise ValueError(f"Unknown score kind: {score_kind}")

    term2 = p_tensor * 2.0 * b_c * torch.tanh(delta / 2.0)
    score = float(term1_weight) * term1 + float(term2_weight) * term2
    return score.masked_fill(~cluster_exists.unsqueeze(0), 0.0)


def _make_condition_scorer(score_kind, term1_weight, term2_weight):
    def scorer(**kwargs):
        return condition_score(
            **kwargs,
            score_kind=score_kind,
            term1_weight=term1_weight,
            term2_weight=term2_weight,
        )

    return scorer


def _weight_tag(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_args():
    parser = add_common_args(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--delta-mode",
        choices=["exact", "range_bound"],
        default="range_bound",
    )
    parser.add_argument("--full-attention-layers", type=int, default=2)
    parser.add_argument("--score-kind", choices=SCORE_KINDS, default="original")
    parser.add_argument("--term1-weight", type=float, default=1.0)
    parser.add_argument("--term2-weight", type=float, default=1.0)
    parser.add_argument(
        "--method-name",
        default=None,
        help="Output subdirectory. Defaults to a name derived from the score.",
    )
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must be > 0")
    if args.full_attention_layers < 0:
        parser.error("--full-attention-layers must be >= 0")
    if args.term1_weight < 0 or args.term2_weight < 0:
        parser.error("--term weights must be >= 0")
    if args.term1_weight == 0 and args.term2_weight == 0:
        parser.error("At least one term weight must be > 0")
    if args.method_name is None:
        term1_tag = (
            "" if args.term1_weight == 1 else f"_t1w{_weight_tag(args.term1_weight)}"
        )
        args.method_name = (
            f"term_ablation_{args.score_kind}{term1_tag}"
            f"_t2w{_weight_tag(args.term2_weight)}"
        )
    return args


def evaluate_sample(ctx, args, settings, pos_list, model_inputs, labels, ref_logits):
    results = {}
    runner_args = SimpleNamespace(
        seq_len=args.seq_len,
        block_size=args.block_size,
        delta_mode=args.delta_mode,
        full_attention_layers=args.full_attention_layers,
    )
    layer_idx_list = list(range(ctx.model_config.num_hidden_layers))
    for eps in settings:
        logits, _patches, budget = rcb.run_for_eps(
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
            f"[{args.method_name}] eps={eps:g}, "
            f"ppl={results[float(eps)]['student_ppl']:.6f}, "
            f"budget={measured_budget:.6f}"
        )
        del logits, _patches
    return results


def main():
    args = parse_args()
    rcb._condition_score_for_blocks = _make_condition_scorer(
        args.score_kind, args.term1_weight, args.term2_weight
    )
    run_multisample(
        args,
        method=args.method_name,
        settings=args.eps,
        evaluate_sample=evaluate_sample,
        setting_label="eps",
    )


if __name__ == "__main__":
    main()
