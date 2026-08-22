"""
Pareto non-dominated-set helpers for the STS2 reward advisor (SA4).

Returns the full non-dominated set over named objectives. This module never
collapses a front into one weighted scalar — preference weights belong to
whoever *selects a point from* the front (``core.reward_eval``), not here.

Objectives may be maximised or minimised. Dilution is a cost (minimise);
tempo, synergy, and resilience are benefits (maximise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence


@dataclass(frozen=True)
class ObjectiveSpec:
    """One axis of a multi-objective comparison.

    Attributes:
        name: Key into each point's objective mapping.
        maximize: True if larger values are better; False if smaller is better.
    """

    name: str
    maximize: bool = True


# The four SA4 advisor axes. Dilution is a penalty (minimise).
ADVISOR_OBJECTIVES: Sequence[ObjectiveSpec] = (
    ObjectiveSpec("tempo", maximize=True),
    ObjectiveSpec("synergy", maximize=True),
    ObjectiveSpec("dilution", maximize=False),
    ObjectiveSpec("resilience", maximize=True),
)


def _signed(value: float, maximize: bool) -> float:
    """Map a raw objective value onto a 'higher is better' scale."""
    return value if maximize else -value


def dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    specs: Sequence[ObjectiveSpec],
) -> bool:
    """True if ``left`` Pareto-dominates ``right`` under ``specs``.

    ``left`` dominates ``right`` when it is at least as good on every
    objective and strictly better on at least one. Equal points do not
    dominate each other.
    """
    if not specs:
        return False
    any_strict = False
    for spec in specs:
        a = _signed(float(left[spec.name]), spec.maximize)
        b = _signed(float(right[spec.name]), spec.maximize)
        if a < b:
            return False
        if a > b:
            any_strict = True
    return any_strict


def non_dominated_indices(
    points: Sequence[Mapping[str, float]],
    specs: Sequence[ObjectiveSpec] = ADVISOR_OBJECTIVES,
) -> List[int]:
    """Return indices of the non-dominated points, in input order.

    An empty input yields an empty result. Points that are mutually equal
    on every objective are all retained (neither dominates the other).
    """
    n = len(points)
    if n == 0:
        return []
    kept: List[int] = []
    for i, candidate in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if dominates(other, candidate, specs):
                dominated = True
                break
        if not dominated:
            kept.append(i)
    return kept
