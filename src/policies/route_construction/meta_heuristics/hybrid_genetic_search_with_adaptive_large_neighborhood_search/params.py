from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
    ALNSParams,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams


@dataclass
class HGSALNSParams:
    """
    Parameters for the Hybrid Genetic Search with ALNS Education metaheuristic.

    Uses ALNS for the education phase and HGS for the routing phase, combining
    the exploration power of genetic algorithms with the exploitation strength
    of adaptive large neighborhood search.
    """

    # Hybrid-specific parameters
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # HGS Components (Genetic Evolution & Population Management)
    hgs_params: HGSParams = field(
        default_factory=lambda: HGSParams(
            time_limit=60.0,
            mu=50,
            nb_elite=10,
            mutation_rate=0.2,
            crossover_rate=0.7,
            n_offspring=40,  # Default for lambda_param
            n_iterations_no_improvement=10,
            nb_granular=10,
            local_search_iterations=500,
            max_vehicles=0,
        )
    )

    # ALNS Components (Education & Intensive Local Search)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            time_limit=60.0,
            max_iterations=50,
            start_temp=100.0,
            cooling_rate=0.995,
            reaction_factor=0.1,
            min_removal=1,
            max_removal_pct=0.3,
        )
    )

    @classmethod
    def from_config(cls, config: Any) -> HGSALNSParams:
        """Create HGSALNSParams from a configuration object."""
        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            hgs_params=HGSParams.from_config(getattr(config, "hgs", config)),
            alns_params=ALNSParams.from_config(getattr(config, "alns", config)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {
            "time_limit": self.time_limit,
            "seed": self.seed,
            "vrpp": self.vrpp,
            "profit_aware_operators": self.profit_aware_operators,
        }
