"""
Greedy Insertion with Blinks Operator Module.

Stochastic version of greedy insertion.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def greedy_blink_insertion(
    build: np.ndarray, budget: float, problem: Optional[BuildProblem], blink_prob: float = 0.1
) -> np.ndarray:
    """
    Greedy insertion with a 'blink' (random bypass) probability.
    to promote exploration in the ALNS repair phase.

    Args:
        build: Current build array.
        budget: Maximum cost allowed.
        problem: BuildProblem context.
        blink_rate: Probability to ignore the best item.

    Returns:
        np.ndarray: Modified build array.
    """
    if problem is None:
        return build

    new_build = build.copy()
    empty_slots = np.where(new_build == -1)[0]

    current_items = new_build[new_build != -1]
    current_cost = np.sum(problem.costs[current_items]) if len(current_items) > 0 else 0.0

    for slot in empty_slots:
        item_scores = []

        for item in range(problem.num_items):
            if item in new_build:
                continue

            item_cost = problem.costs[item]
            if current_cost + item_cost > budget:
                continue

            new_build[slot] = item
            score = problem.evaluate(new_build)
            item_scores.append((score, item))

        new_build[slot] = -1

        if not item_scores:
            continue

        # Sort items by score descending
        item_scores.sort(key=lambda x: x[0], reverse=True)

        # Blink mechanism: iterate through sorted items, blinking with blink_prob
        selected_item = -1
        for _score, item in item_scores:
            if random.random() > blink_prob:
                selected_item = item
                break

        # Fallback to the best if all blinked
        if selected_item == -1:
            selected_item = item_scores[0][1]

        new_build[slot] = selected_item
        current_cost += problem.costs[selected_item]

    return new_build
