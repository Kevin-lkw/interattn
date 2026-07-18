"""Triton condition-block variant with the simplified term-1 softmax.

This is an explicit experimental entry point.  It preserves the production
``condition_block_triton`` implementation and changes only the routing score:

    term1 = 2 B * softmax(z + delta)
    term2 = 2 B_C * softmax(z) * tanh(delta / 2)

The term-1 expression is the ``full_mass_exp`` rule validated by the term
ablation.  It removes the original ``cosh(delta) - 1`` and its normalizer from
the Triton selection path.
"""

from .condition_block_triton_impl.core import (
    build_condition_args,
    generate_condition_block_cached as _generate_condition_block_cached,
    run_prefill_only_condition_block,
)


def generate_condition_block_cached(**kwargs):
    """Generate with the mass-exp term-1 routing score."""
    return _generate_condition_block_cached(**kwargs, term1_mass_exp=True)


__all__ = [
    "build_condition_args",
    "generate_condition_block_cached",
    "run_prefill_only_condition_block",
]
