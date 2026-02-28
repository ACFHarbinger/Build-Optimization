"""
Random Insertion Operator Module.

Inserts items randomly into empty slots.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def random_insertion(build: np.ndarray, budget: float, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Randomly insert items into empty slots until budget is exhausted or no items left.

    Args:
        build: Current build array.
        budget: Maximum cost allowed.
        problem: BuildProblem context.

    Returns:
        np.ndarray: Modified build array.
    """
    if problem is None:
        return build

    new_build = build.copy()
    empty_slots = list(np.where(new_build == -1)[0])
    random.shuffle(empty_slots)

    current_items = set(new_build[new_build != -1])
    available_items = list(set(range(problem.num_items)) - current_items)
    random.shuffle(available_items)

    current_cost = np.sum(problem.costs[list(current_items)]) if current_items else 0.0

    for slot in empty_slots:
        if not available_items:
            break

        # Try items until one fits or we run out
        for i, item in enumerate(available_items):
            if current_cost + problem.costs[item] <= budget:
                new_build[slot] = item
                current_cost += problem.costs[item]
                available_items.pop(i)
                break

    return new_build
