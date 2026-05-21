"""
Run count-all routing with the value-error correction from compare_error.py.

Per layer, this runner replaces each selected head output with value_corrected_v:
    (approx_num + value_num_error) / full_den

This is a diagnostic counterpart to the key-error runner. It follows the current
compare_error.py definition exactly, including the full-attention denominator.
"""

from .runner_inter_error_common import (
    build_error_runner_parser,
    run_error_correction_runner,
)


def parse_args():
    return build_error_runner_parser(
        description=(
            "Run baseline/optimal/inter_value_error routing online and compare PPL. "
            "inter_value_error applies the value-error correction from compare_error.py."
        ),
        method_name="inter_value_error",
    )


def main():
    args = parse_args()
    run_error_correction_runner(
        args=args,
        correction_name="value",
        runner_name="runner_inter_value_error",
        summary_filename="runner_inter_value_error_summary.pt",
    )


if __name__ == "__main__":
    main()
