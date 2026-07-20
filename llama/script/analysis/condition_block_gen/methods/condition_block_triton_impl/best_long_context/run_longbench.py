"""Run LongBench with the best measured condition-block configuration."""

from __future__ import annotations

import argparse
import sys

from ....longbench import run_all
from .config import BEST_LONG_CONTEXT_CONFIG, configured_environment


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--collect-stats",
        action="store_true",
        help="Record realized budgets; disables CUDA graph for quality runs.",
    )
    return parser.parse_known_args(argv)


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Preset wrapper option: --collect-stats records realized budgets and disables CUDA graph.\n")
    wrapper_args, forwarded = parse_wrapper_args(sys.argv[1:])
    config = BEST_LONG_CONTEXT_CONFIG
    # Append fixed method arguments so a forwarded duplicate cannot silently
    # turn the preset into a different algorithm.
    sys.argv = [
        sys.argv[0],
        *forwarded,
        "--method",
        config.method,
        "--condition-block-size",
        str(config.block_size),
        "--condition-eps",
        str(config.eps),
    ]
    with configured_environment(
        cuda_graph=not wrapper_args.collect_stats,
        collect_stats=wrapper_args.collect_stats,
    ):
        run_all.main()


if __name__ == "__main__":
    main()
