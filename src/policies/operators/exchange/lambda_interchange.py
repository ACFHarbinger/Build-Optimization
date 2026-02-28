"""
Lambda Interchange Operator.

Swap up to lambda items between the build and the global pool.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def lambda_interchange(
    build: np.ndarray, problem: Optional[BuildProblem], budget: float, max_lambda: int = 2
) -> np.ndarray:
    """
    Exchange up to 'lambda' items currently in the build with items outside.
    """
    if problem is None:
        return build

    new_build = build.copy()
    filled_slots = list(np.where(new_build != -1)[0])

    current_items = set(new_build[filled_slots])
    available_items = list(set(range(problem.num_items)) - current_items)

    k = random.randint(
        1, min(max_lambda, len(filled_slots), len(available_items)) if len(filled_slots) > 0 and available_items else 0
    )
    if k == 0:
        return new_build

    slots_to_swap = random.sample(filled_slots, k)
    items_to_insert = random.sample(available_items, k)

    for i in range(k):
        new_build[slots_to_swap[i]] = items_to_insert[i]

    current_cost = sum(problem.costs[i] for i in new_build if i != -1)
    if current_cost > budget:
        return build.copy()  # Revert entire operation if over budget

    return new_build
