"""
Greedy Insertion Operator Module.

Inserts items greedily based on the best immediate score improvement.
"""

from typing import Optional

import numpy as np

from core.problem import BuildProblem


def greedy_insertion(build: np.ndarray, budget: float, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Greedy insertion for Build Optimization. Inserts the items that
    provide the best immediate score increase without exceeding the budget.

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
    empty_slots = np.where(new_build == -1)[0]

    # Calculate current cost
    current_items = new_build[new_build != -1]
    current_cost = np.sum(problem.costs[current_items]) if len(current_items) > 0 else 0.0

    # Greedily fill empty slots
    for slot in empty_slots:
        best_score = float("-inf")
        best_item = -1

        # Test all possible items
        for item in range(problem.num_items):
            if item in new_build:
                continue

            item_cost = problem.costs[item]
            if current_cost + item_cost > budget:
                continue

            # Temporarily place item and check score
            new_build[slot] = item
            score = problem.evaluate(new_build)
            if score > best_score:
                best_score = score
                best_item = item

        # Commit the best item if found
        if best_item != -1:
            new_build[slot] = best_item
            current_cost += problem.costs[best_item]
        else:
            # Revert if no item could be placed due to budget constraints
            new_build[slot] = -1

    return new_build
