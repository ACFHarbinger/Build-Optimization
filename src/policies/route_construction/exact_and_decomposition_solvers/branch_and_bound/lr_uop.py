"""
Lagrangian Relaxation + Branch-and-Bound (LR-UOP) solver for VRPP.

Entry point for the "lr_uop" formulation dispatched from dispatcher.py.

Overview
--------
The vehicle capacity constraint  Σ_{i∈A_d} w_i ≤ Q  is Lagrangian-relaxed
with multiplier λ ≥ 0, yielding:

    L(λ) = max_{A_d} [ Σ_{i∈A_d} (r_w - λ)·w_i - c_km·dist(A_d) ] + λ·Q

The inner maximisation is an *uncapacitated* Orienteering Problem (UOP), which is
solvable exactly by Gurobi without capacity bookkeeping. L(λ) is a valid upper
bound on the original VRPP for every λ ≥ 0.

Phase 1 – Subgradient optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starting from λ₀ (configurable, default 0), the Polyak step-size rule is used
to descend toward the minimiser λ* of L(λ):

    g_k  = Q - Σ_{i∈A_k*} w_i      (subgradient)
    α_k  = θ · (L(λ_k) - LB_k) / ‖g_k‖²
    λ_{k+1} = max(0, λ_k - α_k · g_k)

At each iteration, any capacity-feasible OP solution yields a valid lower bound.

Phase 2 – Branch-and-Bound with Lagrangian bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A best-bound-first B&B tree is explored. At each node the Lagrangian bound at
λ* is computed by solving the uncapacitated OP with the node's forced-in /
forced-out customer fixings. If the solution is capacity-feasible the incumbent
is updated; otherwise the heaviest free customer is branched on.

Branching strategies (lr_branching_strategy):
    "max_waste"  – branch on the free OP-selected customer with the highest waste.
                   Rationale: removing the heaviest customer most aggressively
                   reduces capacity violation, recovering feasibility quickly.
    "min_profit" – branch on the free customer with the smallest modified profit
                   (R - λ*)·w_i, minimising the impact on the Lagrangian bound.

When to use
~~~~~~~~~~~
Best suited when:
    - Capacity is the dominant binding constraint (not the tour-structure).
    - The instance has ≤ ~50 candidate customers (OP solves stay fast).
    - LP relaxation bounds (MTZ/DFJ) are historically weak for the instance class.

References:
    Held, M., & Karp, R. M. (1971). The traveling-salesman problem and minimum
    spanning trees: Part II. Mathematical Programming, 1(1), 6–25.

    Fisher, M. L. (1981). The Lagrangian Relaxation Method for Solving Integer
    Programming Problems. Management Science, 27(1), 1–18.
"""

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization import (
    _nearest_neighbour_tour_cost,
    run_subgradient,
)
from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem import (
    solve_uncapacitated_op,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .params import BBParams

# ---------------------------------------------------------------------------
# B&B Node
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _LRNode:
    """
    A node in the Lagrangian Relaxation B&B search tree.

    Attributes:
        bound:      Lagrangian upper bound at this node (for best-bound ordering).
        forced_in:  Customer indices fixed to y_i = 1 at this node.
        forced_out: Customer indices fixed to y_i = 0 at this node.
        depth:      Distance from the root (root = 0).
    """

    bound: float = field(compare=True)
    forced_in: Set[int] = field(compare=False, default_factory=set)
    forced_out: Set[int] = field(compare=False, default_factory=set)
    depth: int = field(compare=False, default=0)


# ---------------------------------------------------------------------------
# Branching helper
# ---------------------------------------------------------------------------


def _select_branch_customer(
    op_visited: Set[int],
    forced_in: Set[int],
    forced_out: Set[int],
    wastes: Dict[int, float],
    R: float,
    lam: float,
    strategy: str,
) -> Optional[int]:
    """
    Select the customer to branch on from the free nodes included in op_visited.

    Free nodes are those in op_visited that are not yet fixed (neither forced-in
    nor forced-out). Branching on a forced-in customer is a no-op (it is already
    selected), so only free customers are considered.

    Args:
        op_visited:  Customers selected by the uncapacitated OP at this node.
        forced_in:   Customers already fixed to y = 1.
        forced_out:  Customers already fixed to y = 0.
        wastes:      Waste per customer.
        R:           Revenue coefficient.
        lam:         Current Lagrange multiplier λ*.
        strategy:    "max_waste" or "min_profit".

    Returns:
        Customer index to branch on, or None if no free candidate exists.
    """
    candidates = [i for i in op_visited if i not in forced_in and i not in forced_out]
    if not candidates:
        return None

    if strategy == "min_profit":
        return min(candidates, key=lambda i: (R - lam) * wastes.get(i, 0.0))
    # Default: "max_waste"
    return max(candidates, key=lambda i: wastes.get(i, 0.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_bb_lr_uop(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: Optional[BBParams] = None,
    mandatory_indices: Optional[Set[int]] = None,
    env: Any = None,
    recorder: Optional[PolicyStateRecorder] = None,
    **kwargs: Any,
) -> Tuple[List[List[int]], float]:
    """
    Solve VRPP using Lagrangian Relaxation + Branch-and-Bound (LR-UOP formulation).

    This is the dispatcher entry point called by ``run_bb_optimizer`` when
    ``params.formulation == "lr_uop"``.  The interface mirrors ``run_bb_mtz``
    and ``run_bb_dfj`` so the dispatcher can treat all three uniformly.

    Args:
        dist_matrix:      Symmetric distance matrix (n × n), index 0 = depot.
        wastes:           {local_customer_id → fill_level} for nodes 1..n-1.
        capacity:         Vehicle capacity Q.
        R:                Revenue coefficient (r_w).
        C:                Distance cost coefficient (c_km).
        params:           BBParams instance; LR-specific fields are prefixed lr_.
        mandatory_indices:  Customers that MUST appear in every feasible solution.
        env:              Optional shared Gurobi environment.
        recorder:         Optional telemetry recorder.

    Returns:
        (routes, objective_value) matching the dispatcher contract:
            routes     – List of routes; each route is a list of customer node IDs.
            objective  – Best feasible objective (revenue - travel cost) found.
    """
    if params is None:
        params = BBParams()

    mandatory = mandatory_indices or set()
    start_time = time.perf_counter()

    # -------------------------------------------------------------------
    # Phase 1: Subgradient optimisation → λ*, UB*, LB_sg
    # -------------------------------------------------------------------
    sg_budget = params.lr_subgradient_time_fraction * params.time_limit

    lam_star, ub_star, lb_sg, _history = run_subgradient(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        mandatory_indices=mandatory,
        params=params,
        time_budget=sg_budget,
        env=env,
        recorder=recorder,
    )

    if recorder:
        recorder.record(
            engine="bb_lr_subgradient",
            lam_star=lam_star,
            ub_star=ub_star,
            lb_sg=lb_sg,
            sg_time=time.perf_counter() - start_time,
        )

    # -------------------------------------------------------------------
    # Phase 2: B&B with Lagrangian bounds at λ*
    # -------------------------------------------------------------------
    incumbent_obj: float = lb_sg
    incumbent_visited: Set[int] = set()

    counter = itertools.count()
    root = _LRNode(bound=ub_star, forced_in=set(mandatory), forced_out=set(), depth=0)
    queue: List[Tuple[float, int, _LRNode]] = [(-ub_star, next(counter), root)]
    nodes_explored = 0
    op_tl = params.lr_op_time_limit

    while queue and nodes_explored < params.lr_max_bb_nodes:
        if time.perf_counter() - start_time >= params.time_limit:
            break

        neg_bound, _, node = heapq.heappop(queue)
        node_bound = -neg_bound
        nodes_explored += 1

        # Prune by inherited bound (before solve)
        gap_threshold = incumbent_obj + abs(incumbent_obj) * params.mip_gap
        if incumbent_obj > -float("inf") and node_bound <= gap_threshold:
            continue

        # Solve uncapacitated OP at λ* with this node's fixings
        op_visited, op_obj, _dist = solve_uncapacitated_op(
            dist_matrix=dist_matrix,
            wastes=wastes,
            lam=lam_star,
            R=R,
            C=C,
            forced_in=node.forced_in,
            forced_out=node.forced_out,
            time_limit=op_tl,
            seed=params.seed,
            env=env,
            recorder=recorder,
        )

        lag_bound = op_obj + lam_star * capacity

        # Prune by computed bound
        if incumbent_obj > -float("inf") and lag_bound <= gap_threshold:
            continue

        # Check capacity feasibility of the OP solution
        collected = sum(wastes.get(i, 0.0) for i in op_visited)

        if collected <= capacity + 1e-9:
            # Feasible: compute true VRPP objective and update incumbent
            feasible_obj = _nearest_neighbour_tour_cost(op_visited, wastes, dist_matrix, R, C)
            if feasible_obj > incumbent_obj:
                incumbent_obj = feasible_obj
                incumbent_visited = set(op_visited)
        else:
            # Over capacity: branch on a free customer from the OP solution
            branch_cust = _select_branch_customer(
                op_visited=op_visited,
                forced_in=node.forced_in,
                forced_out=node.forced_out,
                wastes=wastes,
                R=R,
                lam=lam_star,
                strategy=params.lr_branching_strategy,
            )
            if branch_cust is None:
                continue  # Dead end; no free customer to branch on

            # Left child: exclude branch_cust (y = 0)
            heapq.heappush(
                queue,
                (
                    -lag_bound,
                    next(counter),
                    _LRNode(
                        bound=lag_bound,
                        forced_in=set(node.forced_in),
                        forced_out=set(node.forced_out) | {branch_cust},
                        depth=node.depth + 1,
                    ),
                ),
            )

            # Right child: force branch_cust in (y = 1)
            heapq.heappush(
                queue,
                (
                    -lag_bound,
                    next(counter),
                    _LRNode(
                        bound=lag_bound,
                        forced_in=set(node.forced_in) | {branch_cust},
                        forced_out=set(node.forced_out),
                        depth=node.depth + 1,
                    ),
                ),
            )

    # -------------------------------------------------------------------
    # Reconstruct tour from incumbent visited set
    # -------------------------------------------------------------------
    routes = _visited_to_routes(incumbent_visited, dist_matrix)

    if recorder:
        recorder.record(
            engine="bb_lr_uop",
            nodes_explored=nodes_explored,
            incumbent=incumbent_obj,
            lam_star=lam_star,
            total_time=time.perf_counter() - start_time,
        )

    return routes, max(incumbent_obj, 0.0)


def _visited_to_routes(visited: Set[int], dist_matrix: np.ndarray) -> List[List[int]]:
    """
    Build a single route from the visited set using a nearest-neighbour heuristic.

    LR-UOP uses a single-vehicle model, so the result contains at most one route.

    Args:
        visited:     Customer node indices (local, 1-based).
        dist_matrix: Full distance matrix.

    Returns:
        List containing at most one route.
    """
    if not visited:
        return []

    route: List[int] = []
    remaining = set(visited)
    current = 0
    while remaining:
        nearest = min(remaining, key=lambda j: dist_matrix[current][j])
        route.append(nearest)
        remaining.discard(nearest)
        current = nearest

    return [route]
