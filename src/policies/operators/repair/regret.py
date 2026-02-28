"""
Regret-Based Insertion Operator Module.

Inserts items based on regret (the difference between the best and second best insertions).
"""

from typing import Optional

import numpy as np

from core.problem import BuildProblem


def regret_2_insertion(build: np.ndarray, budget: float, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    2-regret insertion for Build Optimization.
    """
    if problem is None:
        return build
    return regret_k_insertion(build, budget, problem, k=2)


def regret_k_insertion(build: np.ndarray, budget: float, problem: Optional[BuildProblem], k: int) -> np.ndarray:
    """
    Regret-based insertion strategy. Evaluates the difference
    between the best item insertion and the k-th best item.
    """
    if problem is None:
        return build

    new_build = build.copy()
    empty_slots = np.where(new_build == -1)[0]

    current_items = new_build[new_build != -1]
    current_cost = np.sum(problem.costs[current_items]) if len(current_items) > 0 else 0.0

    while len(empty_slots) > 0:
        best_regret = float("-inf")
        best_item_overall = -1
        best_slot_overall = -1

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

            item_scores.sort(key=lambda x: x[0], reverse=True)

            if len(item_scores) == 1:
                regret = item_scores[0][0]
            else:
                k_idx = min(k - 1, len(item_scores) - 1)
                regret = item_scores[0][0] - item_scores[k_idx][0]

            if regret > best_regret:
                best_regret = regret
                best_item_overall = item_scores[0][1]
                best_slot_overall = slot

        if best_item_overall != -1:
            new_build[best_slot_overall] = best_item_overall
            current_cost += problem.costs[best_item_overall]
            empty_slots = np.where(new_build == -1)[0]
        else:
            break

    return new_build
