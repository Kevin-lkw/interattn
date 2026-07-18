"""Run condition-block PPL with the sink and recent blocks always expanded.

The first two layers still use full attention by default.  In every compressed
layer, block 0 and the last causally visible block for each query use token-level
attention.  All intervening blocks use the original condition threshold.
"""

from . import runner_cond_block as base


def parse_args():
    args = base.parse_args(
        description=(
            "Run condition-block PPL while always expanding the first (sink) "
            "and last visible (recent) blocks."
        )
    )
    args.force_first_last_blocks = True
    return args


def main():
    base.main(
        parse_args(),
        output_method="condition_block_sink_recent_runner",
        runner_label="condition-block sink+recent runner",
        summary_filename="runner_cond_block_sink_recent_summary.pt",
    )


if __name__ == "__main__":
    main()
