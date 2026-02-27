"""
Configuration parameters for the Simulated Annealing (SA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAParams:
    """
    Configuration for the SA solver.

    Classic Boltzmann acceptance with geometric cooling schedule.

    Attributes:
        initial_temp: Starting temperature.
        alpha: Geometric cooling factor (T *= alpha each iteration).
        min_temp: Temperature floor below which cooling stops.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    initial_temp: float = 100.0
    alpha: float = 0.995
    min_temp: float = 0.01
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
