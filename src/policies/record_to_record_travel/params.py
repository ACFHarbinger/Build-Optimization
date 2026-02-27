"""
Configuration parameters for the Record-to-Record Travel (RR) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RRParams:
    """
    Configuration for the Record-to-Record Travel solver.

    RR tracks the best solution found so far (the "record") and accepts a
    candidate solution if its objective value is within a tolerance band
    below the record.  The tolerance decays linearly over iterations.

    Attributes:
        tolerance: Initial tolerance as fraction of record profit.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    tolerance: float = 0.05
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
