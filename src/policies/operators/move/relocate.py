"""
Relocate Move Operator.

Moves an item from one slot to another empty slot in the build.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_swap(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Relocate an item to an empty slot.
    """
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]
    empty_slots = np.where(new_build == -1)[0]

    if len(filled_slots) == 0 or len(empty_slots) == 0:
        return new_build

    src = random.choice(filled_slots)
    dst = random.choice(empty_slots)

    new_build[dst] = new_build[src]
    new_build[src] = -1

    return new_build
