"""Backwards-compatible metric exports.

New code should import from :mod:`sfn.evaluation.statistics`.
"""

from .statistics import (
    paired_binary_counts,
    paired_bootstrap_difference,
    percentile_summary,
    wilson_interval,
)


def proportion_ci_wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Legacy Wilson helper accepting a z score instead of confidence."""
    return wilson_interval(successes, n, z=z)


__all__ = [
    "paired_binary_counts",
    "paired_bootstrap_difference",
    "percentile_summary",
    "proportion_ci_wilson",
    "wilson_interval",
]
