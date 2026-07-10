"""
Run count-all routing with the key-error correction from compare_error.py.

Per layer, this runner replaces each selected head output with key_corrected_v:
    (approx_num + key_num_error) / full_den

This isolates the PPL effect of correcting the cluster attention mass while still
using each kept representative's original V.
"""

from .runner_inter_error_common import (
    build_error_runner_parser,
    run_error_correction_runner,
)


def parse_args():
    return build_error_runner_parser(
        description=(
            "Run baseline/optimal/inter_key_error routing online and compare PPL. "
            "inter_key_error applies the key-error correction from compare_error.py."
        ),
        method_name="inter_key_error",
    )


def main():
    args = parse_args()
    run_error_correction_runner(
        args=args,
        correction_name="key",
        runner_name="runner_inter_key_error",
        summary_filename="runner_inter_key_error_summary.pt",
    )


if __name__ == "__main__":
    main()
