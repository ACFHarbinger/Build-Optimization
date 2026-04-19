"""
Hyper-ACO Parameters Module.

This module defines the configuration parameters for the Hyper-Heuristic
Ant Colony Optimization algorithm. It uses a dataclass to store and validate
hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_hyper_heuristic.params import HyperACOParams
    >>> params = HyperACOParams(n_ants=10, alpha=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

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
        q0: Exploitation probability for pseudo-random proportional rule.
        lambda_factor: Monotonic conversion factor for negative improvements (default 1.0001).
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
    q0: float = 0.9
    lambda_factor: float = 1.0001
    operators: List[str] = field(default_factory=lambda: OPERATOR_NAMES.copy())
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if not (0 <= self.q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {self.q0}")

    @classmethod
    def from_config(cls, config: Any) -> HyperACOParams:
        """Create HyperACOParams from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            HyperACOParams instance with values from config.
        """
        return cls(
            n_ants=getattr(config, "n_ants", 10),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.1),
            tau_0=getattr(config, "tau_0", 1.0),
            tau_min=getattr(config, "tau_min", 0.01),
            tau_max=getattr(config, "tau_max", 10.0),
            max_iterations=getattr(config, "max_iterations", 50),
            time_limit=getattr(config, "time_limit", 30.0),
            q0=getattr(config, "q0", 0.9),
            lambda_factor=getattr(config, "lambda_factor", 1.0001),
            operators=getattr(config, "operators", None) or OPERATOR_NAMES.copy(),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )
