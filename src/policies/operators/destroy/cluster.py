"""
Cluster Removal Operator Module.

Removes items that have similar costs, acting as a "cluster" in the stat space.
"""

from typing import Optional

import numpy as np

from core.problem import BuildProblem


def cluster_removal(build: np.ndarray, n_remove: int, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Remove items that have similar costs (clustered together in cost space).

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

    # Select a target slot to form cluster around
    target_slot = np.random.choice(filled_slots)
    target_item = new_build[target_slot]
    target_cost = problem.costs[target_item]

    # Calculate difference in cost for all filled items
    items = new_build[filled_slots]
    costs = problem.costs[items]
    diffs = np.abs(costs - target_cost)

    # Sort by diff and remove the closest ones
    closest_indices = np.argsort(diffs)[:n_remove]
    remove_slots = filled_slots[closest_indices]

    new_build[remove_slots] = -1
    return new_build
