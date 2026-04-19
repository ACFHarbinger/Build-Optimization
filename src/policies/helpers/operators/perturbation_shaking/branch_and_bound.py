"""
Branch-and-Bound Perturbation Operator Module.

This module implements a **destroy-and-repair perturbation** that combines the
``destroy.branch_bound`` and ``repair.branch_bound`` operators into a single
high-quality perturbation move.

Conceptual Foundation
---------------------
Shaw (1998) frames the complete ALNS iteration as:

    repeat
        V_r  = Remove(P)        # Destroy: select visits to remove
        P'   = Reinsert(P, V_r) # Repair: B&B/LDS to find best reinsertion
        P    = AcceptanceCriterion(P, P')
    until termination

This module encapsulates exactly one full ``Remove → Reinsert`` cycle as a
stateless perturbation function, making it suitable for:

    * Integration into an ALNS operator pool as a high-quality repair option.
    * Use as a **perturbation move** in Iterated Local Search (ILS) or
      Iterated Greedy (IG) frameworks (Stützle 2006, Ruiz & Stützle 2007).
    * Escaping local optima in Tabu Search when the neighbourhood is exhausted.

There are two public function pairs:

    * ``bb_perturbation``         — pure distance-minimising variant (CVRP).
    * ``bb_profit_perturbation``  — profit-maximising VRPP variant.

Each pair calls the destroy and repair operators with matching parameter names
so that the perturbation budget (``max_discrepancy``) can be split between the
removal and insertion phases.  By default an equal budget is allocated to each
phase.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.branch_bound import (
    ...     bb_perturbation, bb_profit_perturbation
    ... )
    >>> # Standard CVRP perturbation
    >>> new_routes = bb_perturbation(
    ...     routes, dist_matrix, wastes, capacity,
    ...     n_remove=5, destroy_discrepancy=1, repair_discrepancy=2,
    ... )
    >>> # VRPP profit-aware perturbation
    >>> new_routes = bb_profit_perturbation(
    ...     routes, dist_matrix, wastes, capacity, R=1.0, C=1.0,
    ...     n_remove=5, destroy_discrepancy=1, repair_discrepancy=2,
    ... )
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin.branch_and_bound import (
    bb_profit_removal,
    bb_removal,
)
from logic.src.policies.helpers.operators.recreate_repair.branch_and_bound import (
    bb_insertion,
    bb_profit_insertion,
)


def bb_perturbation(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    n_remove: int,
    destroy_discrepancy: int = 1,
    repair_discrepancy: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_repair_pool: bool = False,
    rng: Optional[Random] = None,
    noise: float = 0.0,
    return_removed: bool = False,
) -> List[List[int]]:
    """Perturb a routing plan via one full B&B destroy-and-repair cycle.

    Executes the complete Shaw (1998) ALNS iteration in a single call:

    1. **Destroy** — ``bb_removal`` selects *n_remove* maximally-disruptive
       nodes using LDS with budget *destroy_discrepancy*.
    2. **Repair** — ``bb_insertion`` re-inserts the removed nodes using an LDS
       tree search with budget *repair_discrepancy*.

    The perturbation always returns a plan in which all originally-routed nodes
    are accounted for (inserted back or, as a fallback, given their own route).
    The returned plan is independent (deep copy) of the input *routes*.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix of shape ``(N+1, N+1)``.
        wastes: Mapping from node index to demand (waste volume).
        capacity: Maximum vehicle load per route.
        n_remove: Number of visits to remove and re-insert.  Typically
            ``0.1 * n_nodes`` to ``0.3 * n_nodes`` per ALNS convention.
        destroy_discrepancy: LDS budget for the *removal* phase.  A budget of
            0 gives deterministic greedy worst-recovery removal; 1–2 is the
            practical operating range.  Defaults to 1.
        repair_discrepancy: LDS budget for the *re-insertion* phase.  A budget
            of 0 gives the same route as Farthest-Insertion greedy; 2–3 is the
            practical range.  Defaults to 2.
        mandatory_nodes: Nodes that must appear in the repaired plan.  Passed
            directly to the repair phase.
        expand_repair_pool: If True, the repair phase considers all currently
            unvisited nodes (not just those removed by the destroy phase).
            Useful for VRPP exploration but may be slow for large instances.
        rng: Random number generator for stochastic tie-breaking during the
            destroy phase.  If None, the destroy is deterministic.
        noise: Standard deviation of Gaussian noise added to recovery-cost
            scores in the destroy phase.  Set to 0 (default) for reproducibility.
        return_removed: Unused; reserved for API compatibility with other
            perturbation operators.  Always ignored.

    Returns:
        List[List[int]]: New routes after one destroy-repair cycle.  Empty
            routes are stripped.  The input *routes* object is NOT mutated.

    Raises:
        ValueError: If *n_remove*, *destroy_discrepancy*, or
            *repair_discrepancy* are negative.
    """
    if n_remove < 0:
        raise ValueError(f"n_remove must be non-negative, got {n_remove}")
    if destroy_discrepancy < 0:
        raise ValueError(f"destroy_discrepancy must be non-negative, got {destroy_discrepancy}")
    if repair_discrepancy < 0:
        raise ValueError(f"repair_discrepancy must be non-negative, got {repair_discrepancy}")

    # Work on an independent copy so the caller's plan is never mutated
    working_routes: List[List[int]] = [list(r) for r in routes if r]

    if not working_routes:
        return working_routes

    # --- Phase 1: Destroy ---
    working_routes, removed_nodes = bb_removal(
        working_routes,
        n_remove=n_remove,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        max_discrepancy=destroy_discrepancy,
        rng=rng,
        noise=noise,
    )

    if not removed_nodes:
        return working_routes

    # --- Phase 2: Repair ---
    working_routes = bb_insertion(
        working_routes,
        removed_nodes=removed_nodes,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        max_discrepancy=repair_discrepancy,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_repair_pool,
    )

    return [r for r in working_routes if r]


def bb_profit_perturbation(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_remove: int,
    destroy_discrepancy: int = 1,
    repair_discrepancy: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_repair_pool: bool = False,
    rng: Optional[Random] = None,
    noise: float = 0.0,
    seed_hurdle_factor: float = 0.5,
    return_removed: bool = False,
) -> List[List[int]]:
    """VRPP profit-maximising destroy-and-repair perturbation.

    Analogous to ``bb_perturbation`` but uses the profit-aware variants of
    both the destroy and repair operators:

    1. **Destroy** — ``bb_profit_removal`` removes *n_remove* nodes with the
       lowest marginal profit contributions, using LDS with budget
       *destroy_discrepancy*.
    2. **Repair** — ``bb_profit_insertion`` re-inserts removed nodes using LDS
       with budget *repair_discrepancy*, maximising aggregate route profit
       (revenue − cost).  Economically unviable routes are pruned.

    Non-mandatory nodes that turn out to be unprofitable after re-insertion are
    permanently dropped, consistent with the VRPP opt-in philosophy.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        n_remove: Number of visits to remove and re-insert.
        destroy_discrepancy: LDS budget for the removal phase.  Defaults to 1.
        repair_discrepancy: LDS budget for the re-insertion phase.  Defaults
            to 2.
        mandatory_nodes: Nodes that must appear in the repaired plan.
        expand_repair_pool: Whether the repair considers all unvisited nodes.
        rng: Random number generator for stochastic noise in the destroy phase.
        noise: Standard deviation of Gaussian noise on profit-contribution
            scores during the destroy phase.  Set to 0 for deterministic mode.
        seed_hurdle_factor: Minimum acceptable profit (as a fraction of new-
            route round-trip cost) for the repair phase to open a new route.
            Mirrors the convention in ``repair.branch_bound.bb_profit_insertion``
            and other profit-aware operators.  Defaults to 0.5.
        return_removed: Unused; reserved for API compatibility.

    Returns:
        List[List[int]]: Updated routes after one destroy-repair cycle.
            Economically weak routes are pruned.  The input *routes* is NOT
            mutated.

    Raises:
        ValueError: If *n_remove*, *destroy_discrepancy*, or
            *repair_discrepancy* are negative.
    """
    if n_remove < 0:
        raise ValueError(f"n_remove must be non-negative, got {n_remove}")
    if destroy_discrepancy < 0:
        raise ValueError(f"destroy_discrepancy must be non-negative, got {destroy_discrepancy}")
    if repair_discrepancy < 0:
        raise ValueError(f"repair_discrepancy must be non-negative, got {repair_discrepancy}")

    working_routes: List[List[int]] = [list(r) for r in routes if r]
    if not working_routes:
        return working_routes

    # --- Phase 1: Profit-guided destroy ---
    working_routes, removed_nodes = bb_profit_removal(
        working_routes,
        n_remove=n_remove,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        max_discrepancy=destroy_discrepancy,
        rng=rng,
        noise=noise,
    )

    if not removed_nodes:
        return working_routes

    # --- Phase 2: Profit-guided repair ---
    working_routes = bb_profit_insertion(
        working_routes,
        removed_nodes=removed_nodes,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        max_discrepancy=repair_discrepancy,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_repair_pool,
        seed_hurdle_factor=seed_hurdle_factor,
    )

    return [r for r in working_routes if r]
