"""
Swap Move Operator.

Swaps two items within the build slots.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_swap(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Swap the positions of two items in the build array.
    """
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]

    if len(filled_slots) < 2:
        return new_build

    slots = random.sample(list(filled_slots), 2)
    new_build[slots[0]], new_build[slots[1]] = new_build[slots[1]], new_build[slots[0]]

    return new_build
