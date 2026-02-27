"""
LAHC Policy Adapter.

Adapts the Late Acceptance Hill-Climbing solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.lahc import LAHCConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.late_acceptance_hill_climbing.params import LAHCParams
from logic.src.policies.late_acceptance_hill_climbing.solver import LAHCSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("lahc")
class LAHCPolicy(BaseRoutingPolicy):
    """
    LAHC policy class.

    Visits bins using Late Acceptance Hill-Climbing with a circular queue
    acceptance criterion.
    """

    def __init__(self, config: Optional[Union[LAHCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LAHCConfig

    def _get_config_key(self) -> str:
        return "lahc"

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
        params = LAHCParams(
            queue_size=int(values.get("queue_size", 50)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = LAHCSolver(
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
