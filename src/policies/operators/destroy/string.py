"""
String Removal Operator Module.

Removes contiguous sequences of item assignments.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def string_removal(build: np.ndarray, n_remove: int, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Remove a contiguous string of assignments.

    Args:
        build: Current build array.
        n_remove: Number of items to remove.
        problem: Optional BuildProblem context (unused here).

    Returns:
        np.ndarray: Modified build array.
    """
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]

    if len(filled_slots) <= n_remove:
        new_build[filled_slots] = -1
        return new_build

    start_idx = random.randint(0, len(filled_slots) - n_remove)
    remove_slots = filled_slots[start_idx : start_idx + n_remove]

    new_build[remove_slots] = -1
    return new_build
