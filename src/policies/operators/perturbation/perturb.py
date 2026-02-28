"""
Perturb Operator.

Applies a minor random mutation.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def perturb(build: np.ndarray, problem: Optional[BuildProblem], budget: float) -> np.ndarray:
    """
    Swap one item in the build with one outside it.
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

    slot_to_vacate = random.choice(filled_slots)
    item_to_insert = random.choice(available_items)
    ejected_item = new_build[slot_to_vacate]

    new_build[slot_to_vacate] = item_to_insert

    if sum(problem.costs[i] for i in new_build if i != -1) > budget:
        new_build[slot_to_vacate] = ejected_item

    return new_build
