"""Small, dependency-light statistics used by evaluation reports."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from statistics import NormalDist, fmean, pstdev


def _finite(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("at least one value is required")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("values must be finite")
    return result


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95, z: float | None = None
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    ``z`` is available for compatibility with callers that already have a
    critical value. Otherwise it is derived from the two-sided ``confidence``.
    """
    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if int(successes) != successes or int(total) != total:
        raise TypeError("successes and total must be integers")
    successes, total = int(successes), int(total)
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return (0.0, 0.0)
    if z is None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentiles must be between 0 and 100")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def percentile_summary(
    values: Iterable[float], *, percentiles: Sequence[float] = (5, 25, 50, 75, 90, 95)
) -> dict[str, float | int]:
    """Return count, mean, extrema, and linearly interpolated percentiles."""
    data = sorted(_finite(values))
    summary: dict[str, float | int] = {
        "count": len(data),
        "mean": fmean(data),
        "std": pstdev(data),
        "min": data[0],
        "max": data[-1],
    }
    for percentile in percentiles:
        p = float(percentile)
        key = f"p{p:g}"
        summary[key] = _quantile(data, p / 100.0)
    return summary


def paired_bootstrap_difference(
    a: Iterable[float],
    b: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = 0,
) -> dict[str, float | int]:
    """Bootstrap the paired mean difference ``a - b``.

    Sampling differences directly preserves the pairing and is substantially
    cheaper than independently resampling both samples.
    """
    left, right = _finite(a), _finite(b)
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = [x - y for x, y in zip(left, right, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(fmean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    return {
        "count": n,
        "difference": fmean(differences),
        "confidence": confidence,
        "ci_low": _quantile(estimates, alpha),
        "ci_high": _quantile(estimates, 1.0 - alpha),
        "resamples": int(resamples),
    }


def paired_binary_counts(a: Iterable[bool], b: Iterable[bool]) -> dict[str, int | float]:
    """Return a paired 2x2 table and an exact two-sided McNemar p-value."""
    left, right = list(a), list(b)
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    if not left:
        raise ValueError("at least one pair is required")
    if any(type(value) is not bool for value in left + right):
        raise TypeError("paired binary samples must contain booleans")
    both = sum(x and y for x, y in zip(left, right, strict=True))
    a_only = sum(x and not y for x, y in zip(left, right, strict=True))
    b_only = sum(not x and y for x, y in zip(left, right, strict=True))
    neither = len(left) - both - a_only - b_only
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "count": len(left),
        "both_success": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither_success": neither,
        "discordant": discordant,
        "success_difference": (a_only - b_only) / len(left),
        "mcnemar_exact_p": p_value,
    }
