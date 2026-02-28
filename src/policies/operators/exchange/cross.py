"""
Cross Exchange Operator.

Swaps segments of the build array.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def cross_exchange(build: np.ndarray, problem: Optional[BuildProblem] = None, max_len: int = 3) -> np.ndarray:
    """
    Swaps two contiguous segments of the build array.
    """
    if problem is None:
        return build
    new_build = build.copy()
    if len(new_build) < 2:
        return new_build

    seg_len = random.randint(1, min(max_len, len(new_build) // 2))
    idx1 = random.randint(0, len(new_build) - seg_len * 2)
    idx2 = random.randint(idx1 + seg_len, len(new_build) - seg_len)

    temp = new_build[idx1 : idx1 + seg_len].copy()
    new_build[idx1 : idx1 + seg_len] = new_build[idx2 : idx2 + seg_len]
    new_build[idx2 : idx2 + seg_len] = temp
    return new_build
