"""
Configuration parameters for the Quantum-Inspired Differential Evolution (QDE) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class QDEParams:
    """
    Configuration parameters for the QDE solver.

    Each candidate solution is represented as a quantum amplitude vector q ∈ [0,1]^N,
    where N is the number of customer nodes. Amplitude values are collapsed to a
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
    delta_theta: float = 0.01 * 3.14159  # Rotation gate step size (e.g., 0.01*pi)
    local_search_iterations: int = 100
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "QDEParams":
        """Create parameters from a configuration object."""
        return cls(
            pop_size=getattr(config, "pop_size", 20),
            F=getattr(config, "F", 0.5),
            CR=getattr(config, "CR", 0.9),
            max_iterations=getattr(config, "max_iterations", 200),
            time_limit=getattr(config, "time_limit", 60.0),
            delta_theta=getattr(config, "delta_theta", 0.01 * 3.14159),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
