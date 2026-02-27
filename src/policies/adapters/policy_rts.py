"""
RTS (Reactive Tabu Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.rts import RTSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.reactive_tabu_search.params import RTSParams
from logic.src.policies.reactive_tabu_search.solver import RTSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("rts")
class RTSPolicy(BaseRoutingPolicy):
    """Reactive Tabu Search policy class."""

    def __init__(self, config: Optional[Union[RTSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RTSConfig

    def _get_config_key(self) -> str:
        return "rts"

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
        params = RTSParams(
            initial_tenure=int(values.get("initial_tenure", 7)),
            min_tenure=int(values.get("min_tenure", 3)),
            max_tenure=int(values.get("max_tenure", 20)),
            tenure_increase=float(values.get("tenure_increase", 1.5)),
            tenure_decrease=float(values.get("tenure_decrease", 0.9)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = RTSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
