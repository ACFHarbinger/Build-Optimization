"""
VNS (Variable Neighborhood Search) Policy Adapter.

Adapts the VNS solver to the agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.vns import VNSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.variable_neighborhood_search.params import VNSParams
from logic.src.policies.variable_neighborhood_search.solver import VNSSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("vns")
class VNSPolicy(BaseRoutingPolicy):
    """
    Variable Neighborhood Search policy class.

    Solves the VRPP by systematically exploring a hierarchy of shaking
    neighborhoods (N_1 ... N_{k_max}) with a local search descent between
    each shaking step.  An improvement resets k to 1; exhausting all
    k_max structures completes one outer iteration.
    """

    def __init__(self, config: Optional[Union[VNSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return VNSConfig

    def _get_config_key(self) -> str:
        return "vns"

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
        params = VNSParams(
            k_max=int(values.get("k_max", 5)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_iterations=int(values.get("local_search_iterations", 20)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = VNSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
