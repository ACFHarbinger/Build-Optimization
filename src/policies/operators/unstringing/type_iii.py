"""
Type III Unstringing Operator.

Removes 3 contiguous items from the build.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def apply_type_iii_unstringing(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Remove 3 contiguous items.
    """
    if problem is None:
        return build
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]

    if len(filled_slots) < 3:
        return new_build

    start = random.randint(0, len(filled_slots) - 3)
    slots_to_remove = filled_slots[start : start + 3]

    new_build[slots_to_remove] = -1
    return new_build
