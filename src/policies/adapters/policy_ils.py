"""
ILS (Iterated Local Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ils import ILSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.iterated_local_search.params import ILSParams
from logic.src.policies.iterated_local_search.solver import ILSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("ils")
class ILSPolicy(BaseRoutingPolicy):
    """Iterated Local Search policy class."""

    def __init__(self, config: Optional[Union[ILSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ILSConfig

    def _get_config_key(self) -> str:
        return "ils"

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
        params = ILSParams(
            n_restarts=int(values.get("n_restarts", 30)),
            inner_iterations=int(values.get("inner_iterations", 20)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            perturbation_strength=float(values.get("perturbation_strength", 0.15)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = ILSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
