"""
HS Policy Adapter.

Adapts the Harmony Search (HS) solver to the agnostic BaseRoutingPolicy
interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hs import HSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.harmony_search.params import HSParams
from logic.src.policies.harmony_search.solver import HSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("hs")
class HSPolicy(BaseRoutingPolicy):
    """
    HS policy class.

    Visits bins using Harmony Search.
    """

    def __init__(self, config: Optional[Union[HSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HSConfig

    def _get_config_key(self) -> str:
        return "hs"

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
        params = HSParams(
            hm_size=int(values.get("hm_size", 10)),
            HMCR=float(values.get("HMCR", 0.9)),
            PAR=float(values.get("PAR", 0.3)),
            max_iterations=int(values.get("max_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = HSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
