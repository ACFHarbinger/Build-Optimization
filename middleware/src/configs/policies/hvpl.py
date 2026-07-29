"""
HVPL Configuration for Hydra.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco_ks import KSparseACOConfig
from .alns import ALNSConfig


@dataclass
class HVPLConfig:
    """
    Configuration for the Hybrid Volleyball Premier League policy.

    HVPL integrates three algorithmic paradigms:
        - ACO: Intelligent population initialization (Phase 1)
        - VPL + HGS: Population evolution with genetic operators (Phase 2)
        - ALNS: Deep local search refinement (Phase 3)
    """

    # VPL Population Parameters
    n_teams: int = 30
    max_iterations: int = 100
    substitution_rate: float = 0.2

    # HGS Evolution Parameters
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 3

    # Integration Parameters
    aco_init_iterations: int = 50  # ACO iterations for initialization

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Nested component configs
    aco: KSparseACOConfig = field(
        default_factory=lambda: KSparseACOConfig(
            n_ants=20,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=1.0,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Only one iteration per construction
            time_limit=60.0,
            local_search=False,  # ALNS handles local search
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )
    alns: ALNSConfig = field(
        default_factory=lambda: ALNSConfig(
            max_iterations=100,
            start_temp=100.0,
            cooling_rate=0.95,
            reaction_factor=0.1,
            min_removal=1,
            max_removal_pct=0.3,
            time_limit=60.0,
        )
    )

    # Common policy fields
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
