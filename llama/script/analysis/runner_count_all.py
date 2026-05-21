"""
Run count-refined H2O-all routing online and report final PPL.

This is the multi-layer PPL runner version of compare_count_all.py. Per layer,
it applies the same count refinement:
    refined_logit = qk + log(C)
which is equivalent to compare_error.py's count_all_v.
"""

from .runner_inter_error_common import (
    build_error_runner_parser,
    run_error_correction_runner,
)


def parse_args():
    return build_error_runner_parser(
        description=(
            "Run count-refined H2O-all routing online and compare PPL. "
            "This is the runner version of compare_count_all.py."
        ),
        method_name="count_all",
    )


def main():
    args = parse_args()
    run_error_correction_runner(
        args=args,
        correction_name="count_all",
        runner_name="runner_count_all",
        summary_filename="runner_count_all_summary.pt",
    )


if __name__ == "__main__":
    main()
