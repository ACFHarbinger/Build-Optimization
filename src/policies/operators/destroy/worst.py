"""
Worst Removal Operator Module.

Removes items with the worst individual stats relative to their cost.
"""

from typing import Optional

import numpy as np

from core.problem import BuildProblem


def worst_removal(build: np.ndarray, n_remove: int, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Removes items from the build that contribute the least efficiently.

    Args:
        build: Current build array.
        n_remove: Number of items to remove.
        problem: BuildProblem context.

    Returns:
        np.ndarray: Modified build array.
    """
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]
    if len(filled_slots) <= n_remove:
        new_build[filled_slots] = -1
        return new_build

    costs = problem.costs[new_build[filled_slots]]
    # Highest cost items are "worst" to keep if we are tight on budget
    worst_indices = np.argsort(costs)[-n_remove:]
    remove_slots = filled_slots[worst_indices]

    new_build[remove_slots] = -1
    return new_build
