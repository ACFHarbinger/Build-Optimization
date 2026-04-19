"""
Pattern and Itinerary Crossover (PAIX).
"""

import random

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual import (
    Individual,
)


def pattern_itinerary_crossover(parent_a: Individual, parent_b: Individual, T: int, N: int) -> Individual:
    """
    Pattern and Itinerary Crossover (PAIX).

    1. Inherits patterns p_i from Parent A or Parent B randomly.
    2. Builds giant tours using Order Crossover (OX) applied to the parents' giant tours,
       filtering nodes to respect the inherited pattern.
    """
    # 1. Pattern Inheritance
    child_patterns = np.zeros(N, dtype=int)
    for i in range(1, N):
        if random.random() < 0.5:
            child_patterns[i] = parent_a.patterns[i]
        else:
            child_patterns[i] = parent_b.patterns[i]

    child_giant_tours = []

    # 2. Daily Giant Tour Order Crossover (OX)
    for t in range(T):
        # Nodes that must be in day t
        active_nodes = set(i for i in range(1, N) if (child_patterns[i] >> t) & 1)

        if not active_nodes:
            child_giant_tours.append(np.array([], dtype=int))
            continue

        ta = parent_a.giant_tours[t]
        tb = parent_b.giant_tours[t]

        # Filter parent tours to only contain the chosen active nodes
        ta_filtered = [node for node in ta if node in active_nodes]
        tb_filtered = [node for node in tb if node in active_nodes]

        length = len(active_nodes)
        if length <= 1:
            child_giant_tours.append(np.array(list(active_nodes), dtype=int))
            continue

        # OX Crossover
        c1 = random.randint(0, length - 1)
        c2 = random.randint(0, length - 1)
        if c1 > c2:
            c1, c2 = c2, c1

        child_tour = [-1] * length

        # Copy swath from A
        for i in range(c1, c2 + 1):
            if i < len(ta_filtered):
                child_tour[i] = ta_filtered[i]

        # Fill rest from B
        idx_b = 0
        for i in range(length):
            if child_tour[i] == -1:
                while idx_b < len(tb_filtered) and tb_filtered[idx_b] in child_tour:
                    idx_b += 1
                if idx_b < len(tb_filtered):
                    child_tour[i] = tb_filtered[idx_b]

        # Filter out any -1 just in case
        child_tour = [n for n in child_tour if n != -1]
        child_giant_tours.append(np.array(child_tour, dtype=int))

    return Individual(patterns=child_patterns, giant_tours=child_giant_tours)
