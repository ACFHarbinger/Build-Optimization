"""
Or-Opt Operator.

Relocates a chain of consecutive slots to another position in the build.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_or_opt(build: np.ndarray, problem: Optional[BuildProblem] = None, max_len: int = 3) -> np.ndarray:
    """
    Relocate a sub-segment of items to another slot position.
    """
    if problem is None:
        return build

    new_build = build.copy()
    if len(new_build) < 3:
        return new_build

    seg_len = random.randint(1, min(max_len, len(new_build) - 1))
    src_idx = random.randint(0, len(new_build) - seg_len)

    # Extract segment
    segment = new_build[src_idx : src_idx + seg_len].copy()

    # Remove from original position
    new_build = np.delete(new_build, slice(src_idx, src_idx + seg_len))

    # Insert at new position
    dst_idx = random.randint(0, len(new_build))
    new_build = np.insert(new_build, dst_idx, segment)

    # Pad or truncate to ensure same size if necessary (should be same)
    return new_build[: len(build)]
