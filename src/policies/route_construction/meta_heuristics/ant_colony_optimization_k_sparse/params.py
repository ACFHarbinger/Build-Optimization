"""
ACO Parameters Module.

This module defines the configuration parameters for the K-Sparse Ant Colony
Optimization algorithm. It uses a dataclass to store and validate hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse.params import KSACOParams
    >>> params = KSACOParams(n_ants=20, rho=0.1)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from logic.src.configs.policies import KSparseACOConfig


@dataclass
class KSACOParams:
    """
    Parameters for K-Sparse Ant Colony Optimization (MMAS_exp variant).

    This parameter set follows the experimental MAX-MIN Ant System (MMAS_exp)
    from Hale (2021), using scale-based precision pruning instead of fixed
    capacity pheromone storage.

    Attributes:
        n_ants: Number of ants per iteration.
        k_sparse: Size of candidate lists (k-nearest neighbors for each node).
        alpha: Pheromone importance exponent.
        beta: Heuristic (distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        scale: Precision parameter for pheromone pruning. Edges within 10^-scale
               of default_value are removed from explicit storage.
        tau_0: Initial pheromone level (if None, computed from nearest neighbor heuristic).
        tau_min: Minimum pheromone level (MMAS lower bound).
        tau_max: Maximum pheromone level (MMAS upper bound).
        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        local_search: Whether to apply local search (2-opt) to solutions.
        local_search_iterations: Number of iterations for local search.
        elitist_weight: Weight for best-so-far solution in pheromone update.
        vrpp: Whether to use VRPP (Vehicle Routing Problem with Profits) mode.
        profit_aware_operators: Whether to use profit-aware feasibility checks.
        seed: Random seed for reproducibility.

    Reference:
        Hale, D. "Investigation of Ant Colony Optimization Implementation
        Strategies For Low-Memory Operating Environments", 2021.
    """

    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    scale: float = 5.0
    tau_0: Optional[float] = None
    tau_min: float = 0.001
    tau_max: float = 10.0
    max_iterations: int = 100
    time_limit: float = 30.0
    local_search: bool = True
    local_search_iterations: int = 100
    elitist_weight: float = 1.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    q0: float = 0.9  # Probability of best-edge selection
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if self.scale < 0:
            raise ValueError(f"Scale parameter must be non-negative, got {self.scale}")

    @classmethod
    def from_config(cls, config: Union[KSparseACOConfig, Dict[str, Any]]) -> KSACOParams:
        """Create KSACOParams from an KSparseACOConfig dataclass or dict.

        Args:
            config: KSparseACOConfig dataclass or dict with solver parameters.

        Returns:
            KSACOParams instance with values from config.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            n_ants=getattr(config, "n_ants", 20),
            k_sparse=getattr(config, "k_sparse", 15),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.1),
            scale=getattr(config, "scale", 5.0),
            tau_0=getattr(config, "tau_0", None),
            tau_min=getattr(config, "tau_min", 0.001),
            tau_max=getattr(config, "tau_max", 10.0),
            max_iterations=getattr(config, "max_iterations", 100),
            time_limit=getattr(config, "time_limit", 30.0),
            local_search=getattr(config, "local_search", True),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            elitist_weight=getattr(config, "elitist_weight", 1.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            q0=getattr(config, "q0", 0.9),
            seed=getattr(config, "seed", 42),
        )
