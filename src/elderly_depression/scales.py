"""Lightweight scoring helpers for depression scales."""

from __future__ import annotations

from typing import Iterable


def _sum_items(items: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in items if x is not None and not _is_missing(x)]
    if not vals:
        return None
    return float(sum(vals))


def _is_missing(x: float | int | None) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def score_cesd10(items: Iterable[float | int | None], positive_indices: set[int] | None = None) -> float | None:
    """CESD-10 score. Items are 0-3; positive_indices are 0-based items reversed."""
    positive_indices = positive_indices or {4, 7}
    vals = list(items)
    scored = []
    for i, v in enumerate(vals):
        if v is None or _is_missing(v):
            continue
        s = float(v)
        if i in positive_indices:
            s = 3 - s
        scored.append(s)
    if len(scored) < 7:
        return None
    return float(sum(scored))


def score_phq9(items: Iterable[float | int | None]) -> float | None:
    """PHQ-9 total score (0-27)."""
    return _sum_items(items)


def score_gds15(items: Iterable[float | int | None], positive_yes: set[int] | None = None) -> float | None:
    """GDS-15. Items are yes(1)/no(0); positive_yes are 0-based items where yes counts."""
    positive_yes = positive_yes or {1, 2, 3, 5, 7, 8, 9, 11, 13, 14}
    vals = list(items)
    scored = []
    for i, v in enumerate(vals):
        if v is None or _is_missing(v):
            continue
        scored.append(float(v) if i in positive_yes else 1.0 - float(v))
    if len(scored) < 10:
        return None
    return float(sum(scored))


def score_hamd24(items: Iterable[float | int | None]) -> float | None:
    """HAMD-24 total score (0-76), labels can be mapped separately."""
    return _sum_items(items)


def severity_level(score: float | None, scale: str) -> str | None:
    if score is None:
        return None
    s = scale.lower()
    if s == "cesd10":
        if score >= 10:
            return "depression_risk"
        return "normal"
    if s == "phq9":
        if score >= 20:
            return "severe"
        if score >= 15:
            return "moderately_severe"
        if score >= 10:
            return "moderate"
        if score >= 5:
            return "mild"
        return "normal"
    if s == "gds15":
        if score >= 10:
            return "severe"
        if score >= 5:
            return "mild"
        return "normal"
    if s == "hamd24":
        if score >= 35:
            return "severe"
        if score >= 20:
            return "moderate"
        if score >= 8:
            return "mild"
        return "normal"
    return None
