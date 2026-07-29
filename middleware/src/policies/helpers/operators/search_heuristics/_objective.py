"""
LKH Objective Module.

Provides the standard TSP cost objective used throughout the
Lin-Kernighan-Helsgaun heuristic and all supporting helper modules.

Public API
----------
get_cost(tour, distance_matrix) -> float
    Return the total tour distance for a closed or open tour.

is_better(c1, c2) -> bool
    Simple cost comparison: True iff c1 strictly beats c2.

Typical usage
-------------
>>> from logic.src.policies.helpers.operators.heuristics._objective import (
...     get_cost, is_better
... )
>>> cost = get_cost(tour, dist)
>>> if is_better(cost, best_cost):
...     best_tour = tour[:]
"""

from __future__ import annotations

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# TSP cost objective
# ---------------------------------------------------------------------------


def get_cost(
    tour: List[int],
    distance_matrix: np.ndarray,
) -> float:
    """
    Evaluate a tour's total distance for the TSP objective.

    Args:
        tour: Closed or open node sequence.
        distance_matrix: (n × n) cost matrix.

    Returns:
        Total tour distance (cost).
    """
    n = len(tour) - 1
    c = 0.0
    for i in range(n):
        c += distance_matrix[tour[i], tour[i + 1]]
    return c


def is_better(c1: float, c2: float) -> bool:
    """
    Simple cost comparison for TSP.

    Returns True iff c1 strictly beats c2.
    """
    return c1 < c2 - 1e-6
