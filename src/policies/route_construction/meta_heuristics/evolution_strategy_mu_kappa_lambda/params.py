from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MuKappaLambdaESParams:
    """
    Configuration parameters for (μ,κ,λ) Evolution Strategy.

    Attributes:
        mu: Number of parents (μ).
        kappa: Maximum age of parents (κ). Parents exceeding this age are discarded.
        lambda_: Number of offspring (λ) generated per generation.
        rho: Number of parents participating in recombination (ρ).
        tau_local: Local learning rate for individual step-size adaptation.
        tau_global: Global learning rate for overall step-size scaling.
        initial_sigma: Initial step size (σ) for mutations (5% of search space recommended).
        recombination_type: Type of recombination ('intermediate' or 'discrete').
        max_iterations: Maximum number of generations.
        time_limit: Maximum CPU time in seconds (0 = no limit).
        min_sigma: Minimum allowed step size to prevent premature convergence.
        max_sigma: Maximum allowed step size to prevent divergence.
        bounds_min: Lower bounds for decision variables (optional).
        bounds_max: Upper bounds for decision variables (optional).
        n_removal: Number of nodes to remove in mutation (for routing problems).
        stagnation_limit: Stagnation threshold for restart (for routing problems).
        local_search_iterations: Local search iterations (for routing problems).
        seed (Optional[int]): Random seed for reproducibility.
        vrpp (bool): Whether the problem is VRP with Profits.
        profit_aware_operators (bool): Whether to use profit-aware insertion/removal.
    """

    mu: int = 15
    kappa: int = 7
    lambda_: int = 100
    rho: int = 2
    tau_local: float = 1.0 / (2.0**0.5)  # τ_local = 1/√(2n) for n-dimensional
    tau_global: float = 1.0 / (2.0 * (2.0**0.5))  # τ_global = 1/(2√n)
    initial_sigma: float = 1.0
    recombination_type: str = "intermediate"  # 'intermediate' or 'discrete'
    max_iterations: int = 1000
    time_limit: float = 300.0  # seconds
    min_sigma: float = 1e-10
    max_sigma: float = 10.0
    bounds_min: Optional[float] = -5.0
    bounds_max: Optional[float] = 5.0
    # Routing-specific parameters
    n_removal: int = 3
    stagnation_limit: int = 10
    local_search_iterations: int = 100
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    def __post_init__(self):
        """Validate parameters."""
        assert self.mu > 0, "μ must be positive"
        assert self.kappa > 0, "κ must be positive"
        assert self.lambda_ >= self.mu, "λ must be >= μ for (μ,κ,λ)-selection"
        assert self.rho >= 1 and self.rho <= self.mu, "ρ must be in [1, μ]"
        assert self.recombination_type in ["intermediate", "discrete"], (
            "recombination_type must be 'intermediate' or 'discrete'"
        )
        assert self.initial_sigma > 0, "initial_sigma must be positive"
        assert self.min_sigma > 0 and self.min_sigma < self.max_sigma, (
            "min_sigma must be positive and less than max_sigma"
        )

    @classmethod
    def from_config(cls, config: Any) -> MuKappaLambdaESParams:
        """Build parameters from a configuration object."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 15),
            kappa=getattr(config, "kappa", 7),
            lambda_=getattr(config, "lambda_", 100),
            rho=getattr(config, "rho", 2),
            tau_local=getattr(config, "tau_local", 1.0 / (2.0**0.5)),
            tau_global=getattr(config, "tau_global", 1.0 / (2.0 * (2.0**0.5))),
            initial_sigma=getattr(config, "initial_sigma", 1.0),
            recombination_type=getattr(config, "recombination_type", "intermediate"),
            max_iterations=getattr(config, "max_iterations", 1000),
            time_limit=getattr(config, "time_limit", 300.0),
            min_sigma=getattr(config, "min_sigma", 1e-10),
            max_sigma=getattr(config, "max_sigma", 10.0),
            bounds_min=getattr(config, "bounds_min", -5.0),
            bounds_max=getattr(config, "bounds_max", 5.0),
            n_removal=getattr(config, "n_removal", 3),
            stagnation_limit=getattr(config, "stagnation_limit", 10),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
