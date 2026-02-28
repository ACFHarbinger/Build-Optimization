"""
Two Opt Star Operator.

Variant of two opt intra.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_2opt_star(build: np.ndarray, problem: Optional[BuildProblem] = None, max_len: int = 6) -> np.ndarray:
    """
    Reverse a longer contiguous segment of slots.
    """
    new_build = build.copy()
    if len(new_build) < 2:
        return new_build

    seg_len = random.randint(2, min(max_len, len(new_build)))
    start_idx = random.randint(0, len(new_build) - seg_len)

    # Shuffle segment randomly
    np.random.shuffle(new_build[start_idx : start_idx + seg_len])
    return new_build
