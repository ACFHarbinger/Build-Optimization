"""
Swap Star Operator.

More intensive version of swap for local search.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_swap_star(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Swap up to 3 items in the build array.
    """
    new_build = build.copy()
    filled_slots = list(np.where(new_build != -1)[0])

    k = min(3, len(filled_slots))
    if k < 2:
        return new_build

    slots = random.sample(filled_slots, k)
    swapped_values = [new_build[s] for s in slots]

    # Rotate the values
    for i in range(k):
        new_build[slots[i]] = swapped_values[(i + 1) % k]

    return new_build
