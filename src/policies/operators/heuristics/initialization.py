"""
Initialization Heuristics Module.

Generates initial valid builds.
"""

import numpy as np

from core.problem import BuildProblem
from policies.operators.repair.greedy import greedy_insertion


def generate_initial_build(problem: BuildProblem, budget: float) -> np.ndarray:
    """
    Generate an initial build greedily.
    """
    empty_build = np.full(problem.num_slots, -1, dtype=np.int32)
    return greedy_insertion(empty_build, budget, problem)
