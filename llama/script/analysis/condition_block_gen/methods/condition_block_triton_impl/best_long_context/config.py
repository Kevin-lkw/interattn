"""Single source of truth for the best measured long-context configuration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BestLongContextConfig:
    method: str = "condition_block_triton"
    block_size: int = 64
    eps: float = 0.1
    mixed_summaries: bool = True
    k_bar_dtype: str = "bfloat16"
    post_prefill_static_cache: bool = True
    cuda_graph: bool = True
    collect_stats: bool = False
    tma_bounds: bool = False

    def to_dict(self):
        return asdict(self)


BEST_LONG_CONTEXT_CONFIG = BestLongContextConfig()


def _environment_updates(*, cuda_graph, collect_stats):
    if cuda_graph and collect_stats:
        raise ValueError("CUDA-graph decode requires condition-block stats to be disabled.")
    config = BEST_LONG_CONTEXT_CONFIG
    return {
        "CONDITION_BLOCK_MIXED_SUMMARIES": "1" if config.mixed_summaries else None,
        "CONDITION_BLOCK_K_BAR_DTYPE": config.k_bar_dtype,
        "CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE": (
            "1" if config.post_prefill_static_cache else None
        ),
        "CONDITION_BLOCK_CUDA_GRAPH": "1" if cuda_graph else None,
        "CONDITION_BLOCK_SKIP_STATS": None if collect_stats else "1",
        # These branches were measured separately and are not part of the best
        # composing configuration.
        "CONDITION_BLOCK_TMA_BOUNDS": None,
        "CONDITION_BLOCK_DENSE_STAGE2": None,
        "CONDITION_BLOCK_COMPACT_SDPA_STAGE2": None,
        "CONDITION_BLOCK_LEGACY_STAGE2": None,
        "CONDITION_BLOCK_EAGER_SELECTION": None,
    }


def apply_process_environment(*, cuda_graph=True, collect_stats=False):
    """Apply the preset environment to the current process."""

    updates = _environment_updates(
        cuda_graph=bool(cuda_graph),
        collect_stats=bool(collect_stats),
    )
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return BEST_LONG_CONTEXT_CONFIG


@contextmanager
def configured_environment(*, cuda_graph=True, collect_stats=False):
    """Temporarily apply the measured best configuration.

    Quality runs should pass ``cuda_graph=False, collect_stats=True`` so that
    realized routing budgets are recorded.  Latency/serving runs should use the
    defaults.
    """

    updates = _environment_updates(
        cuda_graph=bool(cuda_graph),
        collect_stats=bool(collect_stats),
    )
    previous = {name: os.environ.get(name) for name in updates}
    try:
        apply_process_environment(
            cuda_graph=cuda_graph,
            collect_stats=collect_stats,
        )
        yield BEST_LONG_CONTEXT_CONFIG
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def apply_latency_preset(args):
    """Apply the preset to the LongBench-v2 latency argparse namespace."""

    config = BEST_LONG_CONTEXT_CONFIG
    args.condition_block_size = config.block_size
    args.condition_eps = config.eps
    args.skip_stats = True
    args.mixed_summaries = config.mixed_summaries
    args.bf16_k_bar = config.k_bar_dtype == "bfloat16"
    args.tma_bounds = config.tma_bounds
    return args
