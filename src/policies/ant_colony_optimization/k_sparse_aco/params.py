"""
ACO Parameters Module.

This module defines the configuration parameters for the K-Sparse Ant Colony
Optimization algorithm. It uses a dataclass to store and validate hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.k_sparse_aco.params import ACOParams
    >>> params = ACOParams(n_ants=20, rho=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logic.src.configs.policies import ACOConfig


@dataclass
class ACOParams:
    """
    Parameters for K-Sparse Ant Colony Optimization.

    Attributes:
        n_ants: Number of ants per iteration.
        k_sparse: Number of pheromone values to retain per node (candidate list size).
        alpha: Pheromone importance exponent.
        beta: Heuristic (distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        q0: Exploitation probability for pseudo-random proportional rule.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone level (for MMAS bounds).
        tau_max: Maximum pheromone level (for MMAS bounds).
        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        local_search: Whether to apply local search (2-opt) to solutions.
        elitist_weight: Weight for best-so-far solution in pheromone update.
    """

    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    q0: float = 0.9
    tau_0: Optional[float] = None
    tau_min: float = 0.001
    tau_max: float = 10.0
    max_iterations: int = 100
    time_limit: float = 30.0
    local_search: bool = True
    elitist_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if not (0 <= self.q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {self.q0}")

    @classmethod
    def from_config(cls, config: ACOConfig) -> ACOParams:
        """Create ACOParams from an ACOConfig dataclass.

        Args:
            config: ACOConfig dataclass with solver parameters.

        Returns:
            ACOParams instance with values from config.
        """
        return cls(
            n_ants=config.n_ants,
            k_sparse=config.k_sparse,
            alpha=config.alpha,
            beta=config.beta,
            rho=config.rho,
            q0=config.q0,
            tau_0=config.tau_0,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            max_iterations=config.max_iterations,
            time_limit=config.time_limit,
            local_search=config.local_search,
            elitist_weight=config.elitist_weight,
        )
