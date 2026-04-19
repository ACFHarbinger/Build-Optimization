from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
        Individual,
    )


def _get_individual_class() -> type:
    """Lazy import to break circular dependency with meta_heuristics.__init__"""
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

    return Individual


def random_node_inheritance_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Random Node Inheritance Crossover.

    For each unique node across both parents, randomly decides which parent's
    ordering to inherit it from. Nodes assigned to P1 appear in P1's order;
    nodes assigned to P2 appear in P2's order. The child giant tour always
    contains the full union of both parent node sets, preserving the HGS
    genotype length invariant.

    Note: This is NOT PIX (Periodic Crossover with Insertions) from Vidal 2011.
    PIX requires multi-depot periodic structure unavailable in single-depot HGS.

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

    # Fix 11: Add upfront invariant check and remove safety passes.
    if set(p1.giant_tour) != set(p2.giant_tour):
        raise ValueError(
            f"random_node_inheritance_crossover requires both parents to have "
            f"identical node sets. Symmetric difference: "
            f"{set(p1.giant_tour).symmetric_difference(set(p2.giant_tour))}."
        )

    p1_nodes = set(p1.giant_tour)
    p2_nodes = set(p2.giant_tour)
    all_nodes = p1_nodes | p2_nodes

    from_p1: Set[int] = set()
    from_p2: Set[int] = set()

    for node in all_nodes:
        if rng.random() < 0.5:
            from_p1.add(node)
        else:
            from_p2.add(node)

    p1_subseq = [node for node in p1.giant_tour if node in from_p1]
    p2_subseq = [node for node in p2.giant_tour if node in from_p2]

    # Fix 12: Interleave P1 and P2 subsequences to remove geographic ordering bias.
    child_gt = []
    i, j = 0, 0
    take_p1 = rng.random() < 0.5  # random starting parent
    while i < len(p1_subseq) or j < len(p2_subseq):
        if take_p1 and i < len(p1_subseq):
            child_gt.append(p1_subseq[i])
            i += 1
        elif j < len(p2_subseq):
            child_gt.append(p2_subseq[j])
            j += 1
        elif i < len(p1_subseq):
            child_gt.append(p1_subseq[i])
            i += 1
        take_p1 = not take_p1

    # Fix 22: Debug assertion: mandatory nodes must appear in the giant tour.
    assert mandatory_nodes is None or all(n in set(child_gt) for n in mandatory_nodes), (
        f"Crossover produced a giant tour missing mandatory nodes: {set(mandatory_nodes) - set(child_gt)}"
    )

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)
