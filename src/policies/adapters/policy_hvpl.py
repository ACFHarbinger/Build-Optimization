"""
HVPL Policy Adapter.

Adapts the Hybrid Volleyball Premier League (HVPL) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hybrid_volleyball_premier_league.hvpl import HVPLSolver
from logic.src.policies.hybrid_volleyball_premier_league.params import HVPLParams

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from .factory import PolicyRegistry


@PolicyRegistry.register("hvpl")
class HVPLPolicy(BaseRoutingPolicy):
    """
    HVPL policy class.

    Visits pre-selected 'must_go' bins using the population-based HVPL metaheuristic.
    """

    def __init__(self, config: Optional[Union[HVPLConfig, Dict[str, Any]]] = None):
        """Initialize HVPL policy with optional config.

        Args:
            config: HVPLConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HVPLConfig

    def _get_config_key(self) -> str:
        """Return config key for HVPL."""
        return "hvpl"

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
        Run HVPL solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Extract sub-params for ACO and ALNS
        # 'values' contains the flattened config

        aco_cfg = values.get("aco", {})
        alns_cfg = values.get("alns", {})

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

        params = HVPLParams(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 50),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = HVPLSolver(
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
