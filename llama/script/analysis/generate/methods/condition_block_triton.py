"""Compatibility entry point for the Triton condition-block implementation.

The implementation lives in ``condition_block_triton_impl`` so kernels and
runtime code can evolve without turning the methods package into one huge
top-level module.
"""

from .condition_block_triton_impl import (
    build_condition_args,
    generate_condition_block_cached,
    run_prefill_only_condition_block,
)

__all__ = [
    "build_condition_args",
    "generate_condition_block_cached",
    "run_prefill_only_condition_block",
]
