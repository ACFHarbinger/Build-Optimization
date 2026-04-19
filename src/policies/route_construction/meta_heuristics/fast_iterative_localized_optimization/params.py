from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FILOParams:
    """
    Configuration parameters for the FILO solver context.
    Aligned with Accorsi & Vigo (2021).
    """

    time_limit: float = 60.0
    max_iterations: int = 50000
    initial_temperature_factor: float = 10.0
    final_temperature_factor: float = 100.0

    # Random Walk Ruin Parameters
    omega_alpha: float = 0.5  # Probability of jumping to a neighbor route

    # Adaptive shaking meta-strategy parameters
    shaking_lb_factor: float = 0.5
    shaking_ub_factor: float = 2.0
    shaking_lb_intensity: float = 0.5
    shaking_ub_intensity: float = 1.5
    omega_base_multiplier: float = 1.0

    # localized search parameters
    delta_gamma: float = 0.25
    gamma_base: float = 1.0
    gamma_lambda: float = 2.0  # Expansion factor for stagnant nodes

    # Selective Vertex Caching
    svc_size: int = 50

    # Initialization & Algorithm 4
    n_cw: int = 100
    route_min_sa_temp: float = 1.0

    local_search_iterations: int = 500
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> FILOParams:
        """Create FILOParams from a configuration object."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            max_iterations=getattr(config, "max_iterations", 50000),
            initial_temperature_factor=getattr(config, "initial_temperature_factor", 10.0),
            final_temperature_factor=getattr(config, "final_temperature_factor", 100.0),
            omega_alpha=getattr(config, "omega_alpha", 0.5),
            shaking_lb_intensity=getattr(config, "shaking_lb_intensity", 0.5),
            shaking_ub_intensity=getattr(config, "shaking_ub_intensity", 1.5),
            omega_base_multiplier=getattr(config, "omega_base_multiplier", 1.0),
            delta_gamma=getattr(config, "delta_gamma", 0.25),
            gamma_base=getattr(config, "gamma_base", 1.0),
            gamma_lambda=getattr(config, "gamma_lambda", 2.0),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            n_cw=getattr(config, "n_cw", 100),
            svc_size=getattr(config, "svc_size", 50),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
