"""
Hyperparameters for the Augmented Hybrid Volleyball Premier League (AHVPL) algorithm.

Extends the base HVPL framework with Hybrid Genetic Search (HGS) integration
for diversity-driven population management and genetic crossover operators.
"""

from dataclasses import dataclass, field

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..hybrid_genetic_search.params import HGSParams


@dataclass
class AHVPLParams:
    """
    Parameters for the Augmented Hybrid Volleyball Premier League metaheuristic.

    Combines VPL population dynamics, ACO initialization, ALNS local search,
    and HGS diversity management / crossover.
    """

    # VPL Population Dynamics
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0

    # HGS Components (Diversity Management & Crossover)
    hgs_params: HGSParams = field(
        default_factory=lambda: HGSParams(
            elite_size=5,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_vehicles=0,
            population_size=50,
        )
    )

    # ACO Components (Initialization & Global Guidance)
    aco_params: ACOParams = field(
        default_factory=lambda: ACOParams(
            n_ants=10,
            max_iterations=1,
            k_sparse=10,
            rho=0.1,
            local_search=False,
        )
    )

    # ALNS Components (Coaching & Deep Local Search)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=100,
            start_temp=100.0,
            cooling_rate=0.95,
            min_removal=1,
            max_removal_pct=0.2,
        )
    )
