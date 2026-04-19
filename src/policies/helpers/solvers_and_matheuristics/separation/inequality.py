"""
VRPP Valid Inequality Definitions.

Defines the mathematical structures and facet forms for cutting planes used
to strengthen the LP relaxation of the Vehicle Routing Problem with Profits.
"""

from __future__ import annotations

from typing import List, Set

import numpy as np


class Inequality:
    """Base class for valid inequalities."""

    def __init__(self, inequality_type: str, degree_of_violation: float):
        self.type = inequality_type
        self.violation = degree_of_violation
        self.rhs = 0.0  # Right-hand side of inequality
        self.node_set: Set[int] = set()  # Set of nodes involved in the inequality

    def __lt__(self, other):
        """Sort by violation (descending)."""
        return self.violation > other.violation


class PCSubtourEliminationCut(Inequality):
    r"""
    Prize-Collecting Subtour Elimination Constraint (PC-SEC).

    For S ⊂ N \ {0} with 2 ≤ |S| ≤ n - 2:
    ∑_{i,j ∈ S} x[i,j] ≤ |S| - 1

    Equivalently (in cut form):
    ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2  if S is visited (strengthened for VRPP)
    """

    def __init__(
        self, node_set: Set[int], violation: float, facet_form: str = "2.1", node_i: int = -1, node_j: int = -1
    ):
        super().__init__("SEC", violation)
        self.node_set = node_set
        self.facet_form = facet_form
        self.node_i = node_i
        self.node_j = node_j
        self.local_only = False  # Task 7: Default to global, mark True for Form 2.3
        self.rhs = 2.0  # Default for Form 2.1


class CapacityCut(Inequality):
    r"""
    Capacity Inequality (Rounded Capacity Cut).

    For S ⊂ N \ {0}:
    ∑_{(i,j) ∈ δ(S)} x[i,j] ≥ 2 ⌈demand(S) / Q⌉

    This ensures that enough vehicles enter the set S to serve all demand.
    """

    def __init__(self, node_set: Set[int], total_demand: float, capacity: float, violation: float):
        super().__init__("CAPACITY", violation)
        self.node_set = node_set
        self.total_demand = total_demand
        self.min_vehicles = int(np.ceil(total_demand / capacity))
        self.rhs = 2.0 * self.min_vehicles


class CombInequality(Inequality):
    """
    Comb Inequality for TSP/VRP.

    For a handle H and teeth T_1, ..., T_s where |T_j ∩ H| = 1 for all j:
    ∑_{e ∈ E(H)} x_e + ∑_{j=1}^{s} ∑_{e ∈ E(T_j)} x_e ≤ |H| + ∑_{j=1}^{s} |T_j| - (s+1)/2

    Comb inequalities are powerful cuts that strengthen the LP relaxation.
    """

    def __init__(self, handle: Set[int], teeth: List[Set[int]], violation: float):
        super().__init__("COMB", violation)
        self.handle = handle
        self.teeth = teeth
        self.node_set = set(handle).union(*(set(t) for t in teeth))
        self.rhs = len(handle) + sum(len(t) for t in teeth) - (len(teeth) + 1) / 2.0
