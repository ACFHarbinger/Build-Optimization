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


def ordered_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Classical Ordered Crossover (OX1) for fixed-length permutations.

    Preserves absolute positioning of the segment from Parent 1, and
    circular relative positioning of the remaining nodes from Parent 2.
    Requires giant_tour to contain all nodes (visited + unvisited).

    Algorithm:
        1. Select random segment [a, b] from Parent 1
        2. Copy segment into child at EXACT same positions [a, b]
        3. Fill remaining positions by circularly sweeping Parent 2 from position (b+1)
        4. Skip nodes already in the segment

    This maintains:
        - Absolute positioning inheritance from Parent 1 (spatial traits)
        - Relative ordering inheritance from Parent 2 (sequential traits)
        - Fixed-length genotype for all nodes (visited + unvisited)

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.
        mandatory_nodes: List of nodes that MUST be visited.

    Returns:
        Child individual with fixed-length giant tour.
    """
    if rng is None:
        rng = random.Random()

    t1 = p1.giant_tour
    t2 = p2.giant_tour
    n = len(t1)

    # Fix 12 (now moved/renumbered): Upfront assertion for identical node sets.
    if set(t1) != set(t2):
        raise ValueError(
            f"ordered_crossover requires both parents to have identical node sets. "
            f"Symmetric difference: {set(t1).symmetric_difference(set(t2))}."
        )

    # Edge case: Empty or small parents
    if n < 2 or len(t2) < 2:
        return _get_individual_class()(t1[:] if t1 else t2[:], expand_pool=p1.expand_pool)

    # Strict consistency checks
    assert len(t1) == len(t2), f"Parent tours must have equal length: {len(t1)} vs {len(t2)}"

    # 1. Select crossover points
    # Fix 1: Guarantee segment never spans the full tour (when n >= 4).
    if n >= 4:
        a, b = sorted(rng.sample(range(1, n - 1), 2))
    else:
        a, b = sorted(rng.sample(range(n), 2))

    # 2. Initialize child and copy segment from p1 into exact same positions
    child_tour = [-1] * n
    child_tour[a : b + 1] = t1[a : b + 1]
    segment_set = set(t1[a : b + 1])

    # 3. Circularly sweep p2 to fill the remaining positions
    curr_p2 = (b + 1) % n
    curr_child = (b + 1) % n

    # O(n) fill logic
    total_to_fill = n - (b - a + 1)
    filled = 0
    steps = 0
    max_steps = 2 * n  # full circular sweep is at most n steps
    while filled < total_to_fill and steps < max_steps:
        node = t2[curr_p2]
        if node not in segment_set:
            child_tour[curr_child] = node
            curr_child = (curr_child + 1) % n
            filled += 1

        curr_p2 = (curr_p2 + 1) % n
        steps += 1

    # Fix 12 (now moved/renumbered): Replace the silent fallback with an explicit ValueError.
    if -1 in child_tour:
        missing_positions = [i for i, v in enumerate(child_tour) if v == -1]
        raise ValueError(
            f"ordered_crossover: {len(missing_positions)} positions could not be "
            f"filled. Parent node pools are inconsistent. Positions: "
            f"{missing_positions}. "
            f"P1 nodes: {set(t1)}, P2 nodes: {set(t2)}, "
            f"difference: {set(t1).symmetric_difference(set(t2))}."
        )

    # Fix 22: Debug assertion: mandatory nodes must appear in the giant tour.
    assert mandatory_nodes is None or all(node in set(child_tour) for node in mandatory_nodes), (
        f"Crossover produced a giant tour missing mandatory nodes: {set(mandatory_nodes) - set(child_tour)}"
    )

    return _get_individual_class()(giant_tour=child_tour, expand_pool=p1.expand_pool)
