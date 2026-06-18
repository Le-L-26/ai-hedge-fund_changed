"""Continuous, magnitude-aware scoring primitives shared by the quant analysts.

The original agents scored fundamentals with binary cliffs: ``metric > threshold``
collapses to 1/0, so an ROE of 14.9% scored identically to -50% and 15.1%
identically to 80%. All magnitude was discarded at an arbitrary step, and
"confidence" was just the fraction of buckets that agreed — not how strong the
underlying numbers actually were.

These helpers replace that with a smooth transform. Each metric maps to a signed
score in (-1, 1) relative to a threshold *center*; a weighted mean of the
available scores gives a single number whose **sign is the signal** and whose
**magnitude is the conviction**. Missing inputs are dropped (not treated as 0),
and coverage is reported so callers can scale confidence for thin data.

This is a faithful upgrade of the existing logic, not a re-design: the same
metrics and the same threshold centers are used — only the resolution improves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Minimum scale so a near-zero threshold (e.g. a 0% growth center) still yields a
# sane saturation width instead of dividing by ~0.
_SCALE_FLOOR = 1e-6


def scale_from_threshold(threshold: float, fraction: float = 0.6, floor: float = _SCALE_FLOOR) -> float:
    """Default saturation width for a metric, derived from its threshold center.

    ``fraction`` controls how quickly tanh saturates: at ``value - threshold ==
    scale`` the score is tanh(1) ≈ 0.76. A scale of ~60% of the threshold means a
    metric must beat (or miss) its bar by a meaningful margin to approach a full
    ±1, which matches how the old binary bars behaved at the boundary while still
    rewarding magnitude beyond it.
    """
    return max(abs(threshold) * fraction, floor)


def signed_score(
    value: float | None,
    threshold: float,
    scale: float | None = None,
    higher_is_better: bool = True,
) -> float | None:
    """Map ``value`` to a signed score in (-1, 1) around ``threshold``.

    Returns None when ``value`` is missing so it can be excluded from the
    aggregate rather than dragging it toward 0. ``higher_is_better=False`` flips
    the sign for "rich above this" metrics like P/E where exceeding the bar is
    bearish.
    """
    if value is None:
        return None
    if scale is None:
        scale = scale_from_threshold(threshold)
    raw = math.tanh((value - threshold) / scale)
    return raw if higher_is_better else -raw


@dataclass
class Verdict:
    """Outcome of aggregating signed scores into a single directional read."""

    signal: str            # "bullish" | "bearish" | "neutral"
    magnitude: float       # |weighted mean| in [0, 1] — raw conviction
    coverage: float        # fraction of weighted inputs that were available
    confidence: float      # magnitude * coverage, in [0, 1]
    score: float           # signed weighted mean in [-1, 1]


def aggregate(
    scores: list[float | None],
    weights: list[float] | None = None,
    neutral_band: float = 0.05,
) -> Verdict:
    """Weighted mean of available signed scores -> a Verdict.

    - Missing scores are dropped; their weight is removed from the denominator
      and recorded as reduced ``coverage`` (so a partial read is presented with
      proportionally lower confidence rather than silently full conviction).
    - ``neutral_band`` keeps a near-zero net read honestly neutral instead of
      forcing a direction off noise.
    """
    if weights is None:
        weights = [1.0] * len(scores)
    if len(weights) != len(scores):
        raise ValueError("scores and weights must be the same length")

    total_weight = sum(w for w in weights if w > 0)
    used_weight = 0.0
    acc = 0.0
    for s, w in zip(scores, weights):
        if s is None or w <= 0:
            continue
        acc += s * w
        used_weight += w

    coverage = (used_weight / total_weight) if total_weight else 0.0
    net = (acc / used_weight) if used_weight else 0.0

    if net > neutral_band:
        signal = "bullish"
    elif net < -neutral_band:
        signal = "bearish"
    else:
        signal = "neutral"

    magnitude = abs(net)
    return Verdict(
        signal=signal,
        magnitude=magnitude,
        coverage=coverage,
        confidence=magnitude * coverage,
        score=net,
    )
