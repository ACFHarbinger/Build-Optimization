"""
Shaw Removal Operator Module.

Removes items based on a combined relatedness metric (cost and slot proximity).
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def shaw_removal(build: np.ndarray, n_remove: int, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Remove related items based on a weighted similarity function.
    In Build Optimization, relatedness considers both item ID/class and cost.

    Args:
        build: Current build array.
        n_remove: Number of items to remove.
        problem: BuildProblem context.

    Returns:
        np.ndarray: Modified build array.
    """
    new_build = build.copy()
    filled_slots = list(np.where(new_build != -1)[0])

    if len(filled_slots) <= n_remove:
        new_build[filled_slots] = -1
        return new_build

    # Start with a random slot
    removed_slots = [random.choice(filled_slots)]
    filled_slots.remove(removed_slots[0])

    # Incrementally remove items that are "related" to the already removed ones
    while len(removed_slots) < n_remove:
        base_slot = random.choice(removed_slots)
        base_item = new_build[base_slot]
        base_cost = problem.costs[base_item]

        # Calculate relatedness string for remaining items (lower is more related)
        # Relatedness = normalized absolute cost difference + normalized slot distance
        best_relatedness = float("inf")
        best_idx = 0
        best_slot = -1

        for i, slot in enumerate(filled_slots):
            item = new_build[slot]
            cost_diff = abs(problem.costs[item] - base_cost)
            slot_dist = abs(slot - base_slot)

            relatedness = cost_diff * 0.5 + slot_dist * 0.5
            if relatedness < best_relatedness:
                best_relatedness = relatedness
                best_idx = i
                best_slot = slot

        removed_slots.append(best_slot)
        filled_slots.pop(best_idx)

    new_build[removed_slots] = -1
    return new_build
