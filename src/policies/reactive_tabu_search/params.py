"""
Configuration parameters for the Reactive Tabu Search (RTS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RTSParams:
    """
    Configuration for the Reactive Tabu Search solver.

    RTS uses short-term memory (tabu list) to forbid recent moves and
    hash-based cycle detection to adaptively adjust tabu tenure.

    Attributes:
        initial_tenure: Starting tabu tenure.
        min_tenure: Minimum allowed tenure.
        max_tenure: Maximum allowed tenure.
        tenure_increase: Multiplicative factor on cycle detection.
        tenure_decrease: Multiplicative factor on long non-cycling periods.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
