"""
RRT (Record-to-Record Travel) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from src.configs.policies.rrt import RRTConfig
from src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from src.policies.record_to_record_travel.params import RRParams
from src.policies.record_to_record_travel.solver import RRSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("rrt")
class RRTPolicy(BaseRoutingPolicy):
    """Record-to-Record Travel policy class."""

    def __init__(self, config: Optional[Union[RRTConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RRTConfig

    def _get_config_key(self) -> str:
        return "rrt"

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
        params = RRParams(
            tolerance=float(values.get("tolerance", 0.05)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = RRSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
