"""
Hyper-ACO Parameters Module.

This module defines the configuration parameters for the Hyper-Heuristic
Ant Colony Optimization algorithm. It uses a dataclass to store and validate
hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco.params import HyperACOParams
    >>> params = HyperACOParams(n_ants=10, alpha=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from logic.src.configs.policies import ACOConfig

from .hyper_operators import OPERATOR_NAMES


@dataclass
class HyperACOParams:
    """
    Parameters for Hyper-Heuristic ACO.

    Attributes:
        n_ants: Number of ants per iteration.
        alpha: Pheromone importance exponent.
        beta: Heuristic importance exponent.
        rho: Pheromone evaporation rate.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone level (MMAS bounds).
        tau_max: Maximum pheromone level (MMAS bounds).
        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        sequence_length: Length of operator sequence each ant constructs.
        q0: Exploitation probability for pseudo-random proportional rule.
        operators: List of operator names to include in the sequence construction.
    """

    n_ants: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    max_iterations: int = 50
    time_limit: float = 30.0
    sequence_length: int = 5
    q0: float = 0.9
    operators: List[str] = field(default_factory=lambda: OPERATOR_NAMES.copy())

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if not (0 <= self.q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {self.q0}")

    @classmethod
    def from_config(cls, config: ACOConfig) -> HyperACOParams:
        """Create HyperACOParams from an ACOConfig dataclass.

        Args:
            config: ACOConfig dataclass with solver parameters.

        Returns:
            HyperACOParams instance with values from config.
        """
        return cls(
            n_ants=config.n_ants,
            alpha=config.alpha,
            beta=config.beta,
            rho=config.rho,
            tau_0=config.tau_0,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            max_iterations=config.max_iterations,
            time_limit=config.time_limit,
            sequence_length=config.sequence_length,
            q0=config.q0,
            operators=config.operators.copy() if config.operators else OPERATOR_NAMES.copy(),
        )
