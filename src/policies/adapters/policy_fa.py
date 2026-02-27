"""
FA Policy Adapter.

Adapts the Discrete Firefly Algorithm (FA) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.fa import FAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.firefly_algorithm.params import FAParams
from logic.src.policies.firefly_algorithm.solver import FASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("fa")
class FAPolicy(BaseRoutingPolicy):
    """
    FA policy class.

    Visits bins using the Discrete Firefly Algorithm.
    """

    def __init__(self, config: Optional[Union[FAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return FAConfig

    def _get_config_key(self) -> str:
        return "fa"

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
        params = FAParams(
            pop_size=int(values.get("pop_size", 20)),
            beta0=float(values.get("beta0", 1.0)),
            gamma=float(values.get("gamma", 0.1)),
            alpha_profit=float(values.get("alpha_profit", 0.5)),
            beta_will=float(values.get("beta_will", 0.3)),
            gamma_cost=float(values.get("gamma_cost", 0.2)),
            alpha_rnd=float(values.get("alpha_rnd", 0.2)),
            max_iterations=int(values.get("max_iterations", 100)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = FASolver(
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
