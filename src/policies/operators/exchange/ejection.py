"""
Ejection Chain Operator.

Compound displacement.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def ejection_chain(build: np.ndarray, problem: Optional[BuildProblem], budget: float, max_depth: int = 3) -> np.ndarray:
    """
    Perform a chain of ejections and insertions.
    """
    if problem is None:
        return build

    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]
    if len(filled_slots) == 0:
        return new_build

    current_items = set(new_build[filled_slots])
    available_items = list(set(range(problem.num_items)) - current_items)
    if not available_items:
        return new_build

    # We need a max_depth value
    depth = random.randint(1, max_depth)
    for _ in range(depth):
        slot_to_vacate = random.choice(filled_slots)
        item_to_insert = random.choice(available_items)

        # Eject and insert
        ejected_item = new_build[slot_to_vacate]
        new_build[slot_to_vacate] = item_to_insert

        # Maintain available list
        available_items.remove(item_to_insert)
        if ejected_item != -1:
            available_items.append(ejected_item)

        current_cost = sum(problem.costs[i] for i in new_build if i != -1)
        if current_cost > budget:
            # Revert if over budget
            new_build[slot_to_vacate] = ejected_item
            available_items.remove(ejected_item)
            available_items.append(item_to_insert)
            break

    return new_build
