"""
HGS-ALNS Hybrid Policy Adapter.

Adapts the Hybrid HGS-ALNS solver to the common simulator policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSALNSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hgs_alns import HGSALNSSolver
from logic.src.policies.hybrid_genetic_search.params import HGSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hgs_alns")
class HGSALNSPolicy(BaseRoutingPolicy):
    """
    Hybrid HGS-ALNS policy class for the simulator.

    Uses ALNS for the intensive education phase of HGS.
    """

    def __init__(self, config: Optional[Union[HGSALNSConfig, Dict[str, Any]]] = None):
        """Initialize HGS-ALNS policy with optional config.

        Args:
            config: HGSALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS-ALNS hybrid."""
        return "hgs_alns"

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
        Run HGS-ALNS hybrid solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        cfg = self._config
        if cfg is not None:
            params = HGSParams(
                time_limit=cfg.time_limit,
                population_size=cfg.population_size,
                elite_size=cfg.elite_size,
                mutation_rate=cfg.mutation_rate,
                max_vehicles=cfg.max_vehicles,
            )
            alns_iter = cfg.alns_education_iterations
        else:
            params = HGSParams(
                time_limit=values.get("time_limit", 10),
                population_size=values.get("population_size", 50),
                elite_size=values.get("elite_size", 10),
                mutation_rate=values.get("mutation_rate", 0.2),
                max_vehicles=values.get("max_vehicles", 0),
            )
            alns_iter = values.get("alns_education_iterations", 50)

        solver = HGSALNSSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            alns_education_iterations=alns_iter,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
