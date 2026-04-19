"""
Configuration parameters for Hybrid Volleyball Premier League (HVPL) with ACO and ALNS.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
    ALNSParams,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)


@dataclass
class HVPLParams:
    """
    Configuration parameters for Hybrid VPL algorithm integrating ACO and ALNS.

    Architecture:
        Phase 1: ACO Initialization - Intelligent population seeding
        Phase 2: VPL Evolution - Population-based optimization with HGS operators
        Phase 3: ALNS Refinement - Deep local search per solution

    The HVPL combines three algorithmic paradigms:
        - ACO: Pheromone-guided construction for diversity
        - VPL + HGS: Population management with genetic operators
        - ALNS: Adaptive destroy-repair for intensification

    Attributes:
        n_teams: Number of active teams in VPL population.
        max_iterations: Maximum VPL iterations (seasons).
        substitution_rate: Probability of replacing weak teams.
        crossover_rate: Probability of HGS crossover vs mutation.
        mutation_rate: Probability of mutation after crossover.
        elite_size: Number of elite teams preserved.
        aco_init_iterations: ACO iterations for population initialization.
        time_limit: Wall-clock time limit in seconds.
        aco_params: ACO algorithm parameters.
        alns_params: ALNS algorithm parameters.
    """

    # VPL Population Parameters
    n_teams: int = 30
    max_iterations: int = 100
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 3

    # Integration Parameters
    aco_init_iterations: int = 50  # Truncated ACO for initialization

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Sub-algorithm Parameters
    aco_params: Optional[KSACOParams] = field(default_factory=lambda: None)
    alns_params: Optional[ALNSParams] = field(default_factory=lambda: None)

    def __post_init__(self):
        """Initialize sub-algorithm parameters with defaults if not provided."""
        if self.aco_params is None:
            self.aco_params = KSACOParams(
                n_ants=20,
                k_sparse=10,
                alpha=1.0,
                beta=2.0,
                rho=0.1,
                tau_0=None,
                tau_min=0.001,
                tau_max=10.0,
                max_iterations=1,  # Only one iteration per construction phase
                time_limit=60.0,
                local_search=False,  # ALNS handles local search
                local_search_iterations=0,
                elitist_weight=1.0,
                vrpp=self.vrpp,
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            # Sync flags if they were already provided but maybe not consistent
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators

        if self.alns_params is None:
            self.alns_params = ALNSParams(
                max_iterations=100,
                start_temp=100.0,
                cooling_rate=0.95,
                reaction_factor=0.1,
                min_removal=1,
                max_removal_pct=0.3,
                time_limit=60.0,
                vrpp=self.vrpp,
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators

        # Validate constraints
        assert self.n_teams > 0, "n_teams must be positive"
        assert 0 <= self.substitution_rate <= 1, "substitution_rate must be in [0, 1]"
        assert 0 <= self.crossover_rate <= 1, "crossover_rate must be in [0, 1]"
        assert 0 <= self.mutation_rate <= 1, "mutation_rate must be in [0, 1]"
        assert self.elite_size >= 1, "elite_size must be at least 1"

    @classmethod
    def from_config(cls, config: Any) -> HVPLParams:
        """Create HVPLParams from a configuration object."""
        return cls(
            n_teams=getattr(config, "n_teams", 30),
            max_iterations=getattr(config, "max_iterations", 100),
            substitution_rate=getattr(config, "substitution_rate", 0.2),
            crossover_rate=getattr(config, "crossover_rate", 0.8),
            mutation_rate=getattr(config, "mutation_rate", 0.1),
            elite_size=getattr(config, "elite_size", 3),
            aco_init_iterations=getattr(config, "aco_init_iterations", 50),
            time_limit=getattr(config, "time_limit", 300.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            aco_params=KSACOParams.from_config(getattr(config, "aco", {})) if getattr(config, "aco", None) else None,
            alns_params=ALNSParams.from_config(getattr(config, "alns", {})) if getattr(config, "alns", None) else None,
        )
