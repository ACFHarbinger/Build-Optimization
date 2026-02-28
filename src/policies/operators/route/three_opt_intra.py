"""
Three Opt Intra Operator.

Complex reordering within the build.
"""

import random
from typing import Optional

import numpy as np

from core.problem import BuildProblem


def move_3opt_intra(build: np.ndarray, problem: Optional[BuildProblem] = None) -> np.ndarray:
    """
    Randomly pick 3 indices and swap their contents.
    """
    new_build = build.copy()
    if len(new_build) < 3:
        return new_build

    indices = random.sample(range(len(new_build)), 3)
    vals = [new_build[i] for i in indices]

    # Shuffle the values and put them back
    random.shuffle(vals)
    for i, idx in enumerate(indices):
        new_build[idx] = vals[i]

    return new_build
