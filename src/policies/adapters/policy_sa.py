"""
SA (Simulated Annealing) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.sa import SAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.simulated_annealing.params import SAParams
from logic.src.policies.simulated_annealing.solver import SASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sa")
class SAPolicy(BaseRoutingPolicy):
    """Simulated Annealing policy class."""

    def __init__(self, config: Optional[Union[SAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SAConfig

    def _get_config_key(self) -> str:
        return "sa"

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
        params = SAParams(
            initial_temp=float(values.get("initial_temp", 100.0)),
            alpha=float(values.get("alpha", 0.995)),
            min_temp=float(values.get("min_temp", 0.01)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = SASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
