"""Shared Double-P threshold configuration helpers.

The PPL entry points key result rows by ``p1``, so each threshold grid must
use a unique ``p1`` value.  Keeping parsing, validation, and the default grids
here prevents the decode-faithful and full-causal runners from silently
drifting apart.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable


PAPER_P_SETTING = (0.95, 0.70)
DENSE_P_SETTING = (1.00, 1.00)

DEFAULT_DECODE_FAITHFUL_P_SETTINGS = (
    (0.50, 0.10),
    (0.65, 0.15),
    (0.75, 0.20),
    (0.85, 0.30),
    (0.90, 0.50),
    PAPER_P_SETTING,
    DENSE_P_SETTING,
)

DEFAULT_FULL_CAUSAL_P_SETTINGS = (
    (0.10, 0.01),
    (0.20, 0.025),
    (0.30, 0.05),
    (0.40, 0.075),
    *DEFAULT_DECODE_FAITHFUL_P_SETTINGS,
)


def parse_p_setting(raw: str) -> tuple[float, float]:
    """Parse ``P1:P2`` or ``P1,P2`` and enforce Double-P nesting."""

    parts = str(raw).replace(",", ":").split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid Double-P setting {raw!r}; expected P1:P2, "
            "for example 0.95:0.70."
        )
    try:
        p1, p2 = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid Double-P setting {raw!r}."
        ) from exc
    if not 0 < p2 <= p1 <= 1:
        raise argparse.ArgumentTypeError(
            f"Double-P requires 0 < p2 <= p1 <= 1, got {p1}:{p2}."
        )
    return p1, p2


def double_p_setting_key(p1: float, p2: float) -> str:
    """Return the stable key used in PPL summaries and output files."""

    return f"p1={float(p1):g}_p2={float(p2):g}"


def p2_by_unique_p1(
    settings: Iterable[tuple[float, float]],
) -> dict[float, float]:
    """Build the PPL result-key mapping, rejecting duplicate ``p1`` values."""

    mapping: dict[float, float] = {}
    for raw_p1, raw_p2 in settings:
        p1 = float(raw_p1)
        p2 = float(raw_p2)
        if not 0 < p2 <= p1 <= 1:
            raise ValueError(
                "Double-P requires 0 < p2 <= p1 <= 1, "
                f"got {p1:g}:{p2:g}."
            )
        if p1 in mapping:
            raise ValueError(
                "Every Double-P setting must have a unique p1 value; "
                f"found p1={p1:g} more than once."
            )
        mapping[p1] = p2
    return mapping
