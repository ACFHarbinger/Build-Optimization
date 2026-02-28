"""
Lin-Kernighan-Helsgaun (LKH) Operator Wrapper.

Simulates an advanced local search for Build Optimization.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem
from policies.operators.exchange.lambda_interchange import lambda_interchange
from policies.operators.move.swap import move_swap


def run_lkh(build: np.ndarray, problem: Optional[BuildProblem], budget: float, max_iter: int = 50) -> np.ndarray:
    """
    Conceptual Local Search (LKH analog).
    """
    if problem is None:
        return build

    best_build = build.copy()
    best_score = problem.evaluate(best_build)

    for _ in range(max_iter):
        # 50% chance to swap inside build, 50% chance to swap with pool
        if random.random() < 0.5:
            candidate = move_swap(best_build, problem)
            if sum(problem.costs[i] for i in candidate if i != -1) > budget:
                continue
        else:
            candidate = lambda_interchange(best_build, problem, budget, max_lambda=2)

        score = problem.evaluate(candidate)
        if score > best_score:
            best_score = score
            best_build = candidate

    return best_build
