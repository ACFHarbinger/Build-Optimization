"""
Random Removal Operator Module.

This module implements the random removal heuristic for Build Optimization.
"""

from typing import Optional

import numpy as np

from core.problem import BuildProblem


def random_removal(build: np.ndarray, n_remove: int, problem: Optional[BuildProblem]) -> np.ndarray:
    """
    Remove items randomly from the build.

    Args:
        build: Current build array.
        n_remove: Number of items to remove.
        problem: Optional BuildProblem context (unused here).

    Returns:
        np.ndarray: Modified build array.
    """
    new_build = build.copy()
    filled_slots = np.where(new_build != -1)[0]
    if len(filled_slots) > 0:
        n_remove = min(n_remove, len(filled_slots))
        remove_slots = np.random.choice(filled_slots, size=n_remove, replace=False)
        new_build[remove_slots] = -1
    return new_build
