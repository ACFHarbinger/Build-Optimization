"""
Lagrangian Relaxation shared infrastructure for VRPP exact solvers.

Provides the subgradient optimisation loop and the uncapacitated Orienteering
Problem (UOP) inner solver used by both the Branch-and-Bound (LR-UOP formulation)
and the Branch-and-Price-and-Cut pre-pruning integration.

Public API
----------
run_subgradient
    Polyak-step subgradient loop that minimises L(λ) = OP(λ) + λ·Q over λ ≥ 0.
    Returns (lam_star, ub_best, lb_best, history).

solve_uncapacitated_op
    Exact Gurobi solver for the uncapacitated Orienteering Problem at a fixed λ.
    Returns (visited_set, op_objective, dist_cost).

_nearest_neighbour_tour_cost
    Greedy nearest-neighbour tour evaluator used internally by run_subgradient
    to produce feasible lower bounds from capacity-satisfying UOP solutions.
    Exported for callers (e.g. lr_uop.py) that reconstruct tour cost directly.
"""

from .subgradient_optimization import _nearest_neighbour_tour_cost, run_subgradient
from .uncapacitated_orienteering_problem import solve_uncapacitated_op

__all__ = [
    "run_subgradient",
    "solve_uncapacitated_op",
    "_nearest_neighbour_tour_cost",
]
