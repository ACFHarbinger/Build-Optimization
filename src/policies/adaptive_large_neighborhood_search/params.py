"""
Configuration parameters for the Adaptive Large Neighborhood Search (ALNS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import ALNSConfig


@dataclass
class ALNSParams:
    """
    Configuration parameters for the ALNS solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing.
        cooling_rate: Temperature decay factor per iteration.
        reaction_factor: Learning rate for operator weight updates (rho).
        min_removal: Minimum number of nodes to remove.
        max_removal_pct: Maximum percentage of nodes to remove.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3

    @classmethod
    def from_config(cls, config: ALNSConfig) -> ALNSParams:
        """Create ALNSParams from an ALNSConfig dataclass.

        Args:
            config: ALNSConfig dataclass with solver parameters.

        Returns:
            ALNSParams instance with values from config.
        """
        return cls(
            time_limit=config.time_limit,
            max_iterations=config.max_iterations,
            start_temp=config.start_temp,
            cooling_rate=config.cooling_rate,
            reaction_factor=config.reaction_factor,
            min_removal=config.min_removal,
            max_removal_pct=config.max_removal_pct,
        )
