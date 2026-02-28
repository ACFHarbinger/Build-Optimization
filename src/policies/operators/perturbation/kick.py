"""
Kick Perturbation Operator.

Applies a large random perturbation to escape local optima.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def kick(build: np.ndarray, problem: Optional[BuildProblem], budget: float, strength: float = 0.3) -> np.ndarray:
    """
    Destroy and reconstruct a portion of the build randomly.
    """
    if problem is None:
        return build

    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]

    if len(filled_slots) == 0:
        return new_build

    n_remove = max(1, int(len(filled_slots) * strength))
    remove_slots = random.sample(list(filled_slots), n_remove)
    new_build[remove_slots] = -1

    current_items = set(new_build[new_build != -1])
    available_items = list(set(range(problem.num_items)) - current_items)

    for slot in remove_slots:
        if not available_items:
            break

        insert_item = random.choice(available_items)
        if problem.costs[insert_item] + sum(problem.costs[i] for i in new_build if i != -1) <= budget:
            new_build[slot] = insert_item
            available_items.remove(insert_item)

    return new_build
