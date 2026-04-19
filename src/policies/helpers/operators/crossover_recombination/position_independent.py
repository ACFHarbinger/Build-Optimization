from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
        Individual,
    )


def _get_individual_class() -> type:
    """Lazy import to break circular dependency with meta_heuristics.__init__"""
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

    return Individual


def position_independent_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Position Independent Crossover (PIX) for VRP.
    Based on the Uniform Crossover for Permutations (Syswerda, 1991).

    Algorithm:
        1. For each position in the giant tour, randomly decide (50/50)
           whether to inherit the node from Parent 1 at that position.
        2. Fixed positions from Parent 1 are copied into the child.
        3. Remaining positions are filled by the remaining nodes from
           Parent 2, preserving Parent 2's relative ordering.

    This operator focuses on preserving the absolute positions of a subset of
    nodes from one parent and the relative order of the rest from the other.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.
        mandatory_nodes: List of nodes that MUST be visited.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    t1 = p1.giant_tour
    t2 = p2.giant_tour
    n = len(t1)

    if n != len(t2):
        raise ValueError("PIX requires both parents to have identical tour lengths.")

    # Fix 19: Upfront node-set check and explicit error.
    if set(t1) != set(t2):
        raise ValueError(
            f"position_independent_crossover requires both parents to have "
            f"identical node sets. Symmetric difference: "
            f"{set(t1).symmetric_difference(set(t2))}."
        )

    child_gt = [-1] * n
    inherited_from_p1 = set()

    # 1. Randomly select positions to inherit from Parent 1
    for i in range(n):
        if rng.random() < 0.5:
            child_gt[i] = t1[i]
            inherited_from_p1.add(t1[i])

    # 2. Fill remaining positions using Parent 2's relative order
    p2_idx = 0
    for i in range(n):
        if child_gt[i] == -1:
            # Find next node in P2 not already inherited from P1
            while p2_idx < n and t2[p2_idx] in inherited_from_p1:
                p2_idx += 1

            if p2_idx < n:
                child_gt[i] = t2[p2_idx]
                p2_idx += 1

    # Fix 10: Explicitly check for unfilled positions.
    if -1 in child_gt:
        missing = [i for i, v in enumerate(child_gt) if v == -1]
        raise ValueError(
            f"position_independent_crossover: {len(missing)} positions "
            f"unfilled. This indicates a parent node-set inconsistency "
            f"that bypassed the upfront check. Positions: {missing}."
        )

    # Pre-Split sanity check: mandatory nodes are in the visited prefix of
    # child_gt, making them more likely to be assigned routes by Split.
    # The definitive enforcement is in LinearSplit.mandatory_nodes.
    assert mandatory_nodes is None or all(node in set(child_gt) for node in mandatory_nodes), (
        f"Crossover produced a giant tour missing mandatory nodes: {set(mandatory_nodes) - set(child_gt)}"
    )

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)
