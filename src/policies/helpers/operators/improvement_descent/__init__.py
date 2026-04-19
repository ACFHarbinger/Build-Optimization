"""
Improvement Operators Package.

Improvement operators are search mechanisms designed to aggressively
find the true local optimum in the current region of the search space.

This package provides three improvement operators, each callable as a
standalone function on a ``List[List[int]]`` route plan:

Heuristic Local Search (steepest descent)
----------------------------------------------------
Operators that methodically evaluate a dense neighbourhood and apply the
globally best improving move at each iteration, terminating at a strict
local minimum.

* **2-opt** (:func:`two_opt_steepest`, :func:`two_opt_steepest_profit`) —
  Reverses sub-sequences within each route; O(n²) per iteration.

* **Or-opt** (:func:`or_opt_steepest`, :func:`or_opt_steepest_profit`) —
  Relocates chains of 1, 2, or 3 consecutive nodes intra- and inter-route;
  covers the full Or-opt-3 neighbourhood in a single scan.

* **Node Exchange** (:func:`node_exchange_steepest`,
  :func:`node_exchange_steepest_profit`) — Steepest-descent pairwise node
  swap, both within and across routes, with capacity feasibility checking.

VRPP variants additionally accept ``R`` (revenue per unit waste) and ``C``
(cost per unit distance).  All operators return a new ``List[List[int]]``
and never mutate the input *routes* object.

Registry
--------
``IMPROVEMENT_OPERATORS`` maps string keys to their corresponding
functions for use in ALNS / ILS operator pools.  CVRP operators are keyed
directly; VRPP operators carry a ``_PROFIT`` suffix.

Example:
    >>> from logic.src.policies.helpers.operators.improvement import (
    ...     two_opt_steepest, or_opt_steepest, node_exchange_steepest
    ... )
    >>> improved = two_opt_steepest(routes, dist_matrix, wastes, capacity)
    >>> improved = or_opt_steepest(routes, dist_matrix, wastes, capacity, C=0.5)
    >>> improved = node_exchange_steepest(routes, dist_matrix, wastes, capacity)
"""

from .node_exchange_steepest import node_exchange_steepest, node_exchange_steepest_profit
from .or_opt_steepest import or_opt_steepest, or_opt_steepest_profit
from .steepest_two_opt import two_opt_steepest, two_opt_steepest_profit

# ---------------------------------------------------------------------------
# Operator registry (CVRP and VRPP variants)
# ---------------------------------------------------------------------------

IMPROVEMENT_OPERATORS = {
    # Heuristic steepest-descent
    "2OPT": two_opt_steepest,
    "2OPT_PROFIT": two_opt_steepest_profit,
    "OR_OPT": or_opt_steepest,
    "OR_OPT_PROFIT": or_opt_steepest_profit,
    "NODE_SWAP": node_exchange_steepest,
    "NODE_SWAP_PROFIT": node_exchange_steepest_profit,
}

IMPROVEMENT_NAMES = list(IMPROVEMENT_OPERATORS.keys())

__all__ = [
    "IMPROVEMENT_NAMES",
    "IMPROVEMENT_OPERATORS",
    "two_opt_steepest",
    "two_opt_steepest_profit",
    "or_opt_steepest",
    "or_opt_steepest_profit",
    "node_exchange_steepest",
    "node_exchange_steepest_profit",
]
