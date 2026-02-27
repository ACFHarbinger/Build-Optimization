"""
ABC Policy Adapter.

Adapts the Artificial Bee Colony (ABC) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.abc import ABCConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.artificial_bee_colony.params import ABCParams
from logic.src.policies.artificial_bee_colony.solver import ABCSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("abc")
class ABCPolicy(BaseRoutingPolicy):
    """
    ABC policy class.

    Visits bins using the Artificial Bee Colony algorithm.
    """

    def __init__(self, config: Optional[Union[ABCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ABCConfig

    def _get_config_key(self) -> str:
        return "abc"

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
        params = ABCParams(
            n_sources=int(values.get("n_sources", 20)),
            limit=int(values.get("limit", 10)),
            max_iterations=int(values.get("max_iterations", 200)),
            n_removal=int(values.get("n_removal", 1)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = ABCSolver(
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
