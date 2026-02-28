"""
Type I Unstringing Operator.

Removes 1 item from the build.
Analogous to a 1-item string removal.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def apply_type_i_unstringing(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Remove 1 item randomly.
    """
    if problem is None:
        return build
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]

    if len(filled_slots) < 1:
        return new_build

    idx = random.choice(filled_slots)
    new_build[idx] = -1
    return new_build
