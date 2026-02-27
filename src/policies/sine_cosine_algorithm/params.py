"""
Configuration parameters for the Sine Cosine Algorithm (SCA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SCAParams:
    """
    Configuration parameters for the SCA solver.

    Positions are updated using trigonometric sine/cosine functions.  A
    control parameter `a` decays from `a_max` to 0, shifting behaviour from
    global exploration (|sin/cos| > 1) to local exploitation (|sin/cos| < 1).
    The continuous position is binarised via sigmoid and decoded to a routing
    solution using the Largest Rank Value (LRV) rule.

    Attributes:
        pop_size: Population size.
        a_max: Initial value of the control parameter (decays to 0).
        max_iterations: Maximum SCA iterations.
        time_limit: Wall-clock time limit in seconds.
    """

    pop_size: int = 20
    a_max: float = 2.0
    max_iterations: int = 200
    time_limit: float = 60.0
