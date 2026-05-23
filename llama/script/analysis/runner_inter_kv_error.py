"""
Run count-all routing with both key-error and value-error corrections.

Per layer, this runner replaces each selected head output with reconstructed_v:
    (approx_num + key_num_error + value_num_error) / full_den

For a fixed layer input this should match full attention V up to numerical
precision.  As a multi-layer runner it is mainly a sanity check that the online
patch/capture path is consistent.
"""

from .runner_inter_error_common import (
    build_error_runner_parser,
    run_error_correction_runner,
)


def parse_args():
    return build_error_runner_parser(
        description=(
            "Run baseline/optimal/inter_kv_error routing online and compare PPL. "
            "inter_kv_error applies both key and value corrections from compare_error.py."
        ),
        method_name="inter_kv_error",
    )


def main():
    args = parse_args()
    run_error_correction_runner(
        args=args,
        correction_name="kv",
        runner_name="runner_inter_kv_error",
        summary_filename="runner_inter_kv_error_summary.pt",
    )


if __name__ == "__main__":
    main()
