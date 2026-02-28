"""
Hyper-Heuristic Operators for Build Optimization.
"""

import random
from typing import Callable, Dict

import numpy as np

from core.problem import BuildProblem


class HyperOperatorContext:
    """
    Context object for HH-ACO operators in Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        build: np.ndarray,
    ):
        self.problem = problem
        self.budget = budget
        self.build = build.copy()
        self.score = problem.evaluate(self.build)

    def update_if_improved(self, new_build: np.ndarray) -> bool:
        """Update current build if the new one is feasible and better/equal."""
        if not self.problem.is_feasible(new_build):
            return False

        new_score = self.problem.evaluate(new_build)
        if new_score >= self.score:
            self.build = new_build
            self.score = new_score
            return True
        return False


def apply_swap(ctx: HyperOperatorContext) -> bool:
    """Swap items between two random slots."""
    if ctx.problem.num_slots < 2:
        return False

    new_build = ctx.build.copy()
    idx1, idx2 = random.sample(range(ctx.problem.num_slots), 2)
    new_build[idx1], new_build[idx2] = new_build[idx2], new_build[idx1]

    return ctx.update_if_improved(new_build)


def apply_greedy_improve(ctx: HyperOperatorContext) -> bool:
    """Greedily improve a random slot."""
    slot_idx = random.randint(0, ctx.problem.num_slots - 1)
    current_item = ctx.build[slot_idx]

    # Current budget excluding this slot
    temp_build = ctx.build.copy()
    temp_build[slot_idx] = -1
    remaining_budget = ctx.budget - ctx.problem.budget_used(temp_build)

    # Candidates for this slot
    candidates = np.where(ctx.problem.slot_ids == slot_idx)[0]
    affordable = candidates[ctx.problem.costs[candidates] <= remaining_budget]

    if len(affordable) == 0:
        return False

    # Find best item for this slot
    score_vec = (ctx.problem.stat_matrix @ ctx.problem.stat_weights) + ctx.problem.rarities * ctx.problem.rarity_bonus
    best_item = affordable[int(np.argmax(score_vec[affordable]))]

    if best_item == current_item:
        return False

    new_build = ctx.build.copy()
    new_build[slot_idx] = best_item
    return ctx.update_if_improved(new_build)


def apply_random_perturb(ctx: HyperOperatorContext) -> bool:
    """Randomly change one slot to a different affordable item."""
    slot_idx = random.randint(0, ctx.problem.num_slots - 1)

    temp_build = ctx.build.copy()
    temp_build[slot_idx] = -1
    remaining_budget = ctx.budget - ctx.problem.budget_used(temp_build)

    candidates = np.where(ctx.problem.slot_ids == slot_idx)[0]
    affordable = candidates[ctx.problem.costs[candidates] <= remaining_budget]

    if len(affordable) <= 1:
        return False

    new_item = int(random.choice(affordable))
    new_build = ctx.build.copy()
    new_build[slot_idx] = new_item
    return ctx.update_if_improved(new_build)


HYPER_OPERATORS: Dict[str, Callable[[HyperOperatorContext], bool]] = {
    "swap": apply_swap,
    "greedy_improve": apply_greedy_improve,
    "random_perturb": apply_random_perturb,
}

OPERATOR_NAMES = list(HYPER_OPERATORS.keys())
