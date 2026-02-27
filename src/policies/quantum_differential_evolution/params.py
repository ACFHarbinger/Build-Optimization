"""
Configuration parameters for the Quantum-Inspired Differential Evolution (QDE) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QDEParams:
    """
    Configuration parameters for the QDE solver.

    Each candidate solution is represented as a quantum amplitude vector q ∈ [0,1]^N,
    where N is the number of customer nodes.  Amplitude values are collapsed to a
    discrete routing solution by ranking and greedy insertion.

    Attributes:
        pop_size: Number of individuals in the population.
        F: Differential mutation scaling factor.
        CR: Binomial crossover rate.
        max_iterations: Maximum number of DE generations.
        time_limit: Hard wall-clock time limit in seconds.
        n_removal: Number of nodes removed per perturbation during collapse repair.
    """

    pop_size: int = 20
    F: float = 0.5
    CR: float = 0.9
    max_iterations: int = 200
    time_limit: float = 60.0
    n_removal: int = 2
