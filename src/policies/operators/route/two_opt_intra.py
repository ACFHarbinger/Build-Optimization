"""
Two Opt Intra Operator.

Reverses a segment of the build array.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_2opt_intra(build: np.ndarray, problem: Optional[BuildProblem] = None, max_len: int = 4) -> np.ndarray:
    """
    Reverse a contiguous segment of slots.
    """
    new_build = build.copy()
    if len(new_build) < 2:
        return new_build

    seg_len = random.randint(2, min(max_len, len(new_build)))
    start_idx = random.randint(0, len(new_build) - seg_len)

    new_build[start_idx : start_idx + seg_len] = new_build[start_idx : start_idx + seg_len][::-1]
    return new_build
