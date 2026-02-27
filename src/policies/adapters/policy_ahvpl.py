"""
AHVPL Policy Adapter.

Adapts the Augmented Hybrid Volleyball Premier League (AHVPL) logic
to the agnostic policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ahvpl import AHVPLConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.augmented_hybrid_volleyball_premier_league.ahvpl import (
    AHVPLSolver,
)
from logic.src.policies.augmented_hybrid_volleyball_premier_league.params import (
    AHVPLParams,
)

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..hybrid_genetic_search.params import HGSParams
from .factory import PolicyRegistry


@PolicyRegistry.register("ahvpl")
class AHVPLPolicy(BaseRoutingPolicy):
    """
    AHVPL policy class.

    Visits pre-selected 'must_go' bins using the augmented HVPL metaheuristic
    combining ACO, VPL, HGS, and ALNS.
    """

    def __init__(self, config: Optional[Union[AHVPLConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return AHVPLConfig

    def _get_config_key(self) -> str:
        return "ahvpl"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run AHVPL solver.

        Returns:
            Tuple of (routes, profit, solver_cost).
        """
        aco_cfg = values.get("aco", {})
        alns_cfg = values.get("alns", {})
        hgs_cfg = values.get("hgs", {})

        aco_params = ACOParams(
            n_ants=aco_cfg.get("n_ants", 10),
            max_iterations=aco_cfg.get("max_iterations", 1),
            k_sparse=aco_cfg.get("k_sparse", 10),
            rho=aco_cfg.get("rho", 0.1),
            local_search=aco_cfg.get("local_search", False),
        )

        alns_params = ALNSParams(
            max_iterations=alns_cfg.get("max_iterations", 100),
            start_temp=alns_cfg.get("start_temp", 100.0),
            cooling_rate=alns_cfg.get("cooling_rate", 0.95),
            min_removal=alns_cfg.get("min_removal", 1),
            max_removal_pct=alns_cfg.get("max_removal_pct", 0.2),
        )

        hgs_params = HGSParams(
            time_limit=hgs_cfg.get("time_limit", values.get("time_limit", 60.0)),
            population_size=hgs_cfg.get("population_size", 50),
            elite_size=hgs_cfg.get("elite_size", 5),
            mutation_rate=hgs_cfg.get("mutation_rate", 0.2),
            crossover_rate=hgs_cfg.get("crossover_rate", 0.7),
            n_generations=hgs_cfg.get("n_generations", 100),
            max_vehicles=hgs_cfg.get("max_vehicles", 0),
        )

        params = AHVPLParams(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 50),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            hgs_params=hgs_params,
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = AHVPLSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
