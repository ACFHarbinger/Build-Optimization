"""
Subgradient optimisation for the Lagrangian multiplier λ.

The Lagrangian relaxation of the capacity constraint yields the dual function:

    L(λ) = max_{A_d} [ Σ_{i∈A_d} (r_w - λ)·w_i - c_km·dist(A_d) ] + λ·Q

L(λ) is convex and piecewise-linear in λ. Its minimum over λ ≥ 0 (the tightest
valid upper bound on the primal VRPP) is found via subgradient optimisation using
the Polyak step-size rule (Held & Karp, 1971):

    g_k  = Q - Σ_{i∈A_k*} w_i          (subgradient at λ_k)
    α_k  = θ · (L(λ_k) - LB_k) / ‖g_k‖²   (Polyak step size)
    λ_{k+1} = max(0, λ_k - α_k · g_k)

where LB_k is the best capacity-feasible objective known at iteration k and
θ ∈ (0, 2] controls step aggressiveness.

Convergence criteria (any met → stop):
    - ‖g_k‖² < ε  (λ already optimal; subgradient is zero)
    - (L(λ_k) - LB_k) / max(|LB_k|, 1) ≤ mip_gap  (gap closed)
    - Iteration budget exhausted
    - Wall-clock time budget exhausted

References:
    Held, M., & Karp, R. M. (1971). The traveling-salesman problem and minimum
    spanning trees: Part II. Mathematical Programming, 1(1), 6–25.

    Fisher, M. L. (1981). The Lagrangian Relaxation Method for Solving Integer
    Programming Problems. Management Science, 27(1), 1–18.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .uncapacitated_orienteering_problem import solve_uncapacitated_op


def _nearest_neighbour_tour_cost(
    visited: Set[int],
    wastes: Dict[int, float],
    dist_matrix: np.ndarray,
    R: float,
    C: float,
) -> float:
    """
    Compute the VRPP objective for a visited set using a nearest-neighbour tour.

    Used to obtain a feasible lower bound (LB) when the OP solution satisfies
    capacity. The tour is approximated greedily; the result is a valid (though
    not necessarily optimal) lower bound.

    Args:
        visited: Customer indices in the candidate solution.
        wastes: {customer_id → fill_level}.
        dist_matrix: Full distance matrix.
        R: Revenue per unit waste.
        C: Cost per unit distance.

    Returns:
        Σ R·w_i - C·dist(nearest-neighbour tour).
    """
    if not visited:
        return 0.0

    revenue = sum(R * wastes.get(i, 0.0) for i in visited)

    route = [0]
    remaining = set(visited)
    current = 0
    while remaining:
        nearest = min(remaining, key=lambda j: dist_matrix[current][j])
        route.append(nearest)
        remaining.discard(nearest)
        current = nearest
    route.append(0)

    dist_cost = sum(dist_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1))
    return revenue - C * dist_cost


def run_subgradient(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_indices: Optional[Set[int]] = None,
    params: Optional[Any] = None,
    time_budget: float = 60.0,
    env: Any = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    """
    Run subgradient optimisation to find λ* that minimises L(λ).

    Iteratively solves the uncapacitated OP inner problem and updates λ via
    the Polyak rule until convergence or the time/iteration budget is consumed.

    Args:
        dist_matrix: Symmetric distance matrix (n × n), index 0 = depot.
        wastes: {customer_id → fill_level} for nodes 1..n-1.
        capacity: Vehicle capacity Q.
        R: Revenue coefficient (r_w).
        C: Cost coefficient (c_km).
        mandatory_indices: Customers that must be visited in every feasible solution.
        params: Any policy parameters instance carrying LR-specific fields.
        time_budget: Wall-clock seconds available for this phase.
        env: Optional shared Gurobi environment.
        recorder: Optional telemetry recorder.

    Returns:
        (lam_best, ub_best, lb_best, history) where:
            lam_best – λ* that produced the tightest Lagrangian bound.
            ub_best  – Tightest Lagrangian upper bound L(λ*).
            lb_best  – Best feasible (capacity-satisfying) objective seen.
            history  – Per-iteration dicts with keys 'lam', 'ub', 'lb', 'g'.
    """
    if params is None:
        raise ValueError("Policy parameters must be provided.")

    mandatory = mandatory_indices or set()
    lam = float(params.lr_lambda_init)
    theta = params.lr_subgradient_theta
    op_tl = params.lr_op_time_limit
    mip_gap = params.mip_gap

    ub_best = float("inf")
    lam_best = lam
    lb_best = -float("inf")
    history: List[Dict[str, float]] = []

    t0 = time.perf_counter()

    for _k in range(params.lr_max_subgradient_iters):
        if time.perf_counter() - t0 >= time_budget:
            break

        visited, op_obj, _dist = solve_uncapacitated_op(
            dist_matrix=dist_matrix,
            wastes=wastes,
            lam=lam,
            R=R,
            C=C,
            forced_in=mandatory,
            time_limit=op_tl,
            seed=params.seed,
            env=env,
            recorder=recorder,
        )

        # Lagrangian bound: L(λ) = OP_obj + λ·Q
        lag_bound = op_obj + lam * capacity
        if lag_bound < ub_best:
            ub_best = lag_bound
            lam_best = lam

        # Subgradient: g = Q - Σ_{i∈A*} w_i
        collected = sum(wastes.get(i, 0.0) for i in visited)
        g = capacity - collected  # positive → slack capacity, negative → infeasible

        # Update LB if OP solution is capacity-feasible
        if collected <= capacity + 1e-9:
            feasible_obj = _nearest_neighbour_tour_cost(visited, wastes, dist_matrix, R, C)
            if feasible_obj > lb_best:
                lb_best = feasible_obj

        history.append({"lam": lam, "ub": lag_bound, "lb": lb_best, "g": g})

        # Convergence: gap closed
        if lb_best > -float("inf") and (ub_best - lb_best) <= mip_gap * max(abs(lb_best), 1.0):
            break

        # Convergence: zero subgradient
        g_sq = g * g
        if g_sq < 1e-12:
            break

        # Polyak step size
        ref_lb = lb_best if lb_best > -float("inf") else 0.0
        gap = max(lag_bound - ref_lb, 1e-8)
        alpha = theta * gap / g_sq
        lam = max(0.0, lam - alpha * g)

    return lam_best, ub_best, lb_best, history
