"""
PSOMA Policy Adapter.

Adapts the Particle Swarm Optimization Memetic Algorithm (PSOMA) to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.psoma import PSOMAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.particle_swarm_optimization_memetic.params import PSOMAParams
from logic.src.policies.particle_swarm_optimization_memetic.solver import PSOMAsSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("psoma")
class PSOMAPolicy(BaseRoutingPolicy):
    """
    PSOMA policy class.

    Visits bins using Particle Swarm Optimization with a memetic local-search step.
    """

    def __init__(self, config: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PSOMAConfig

    def _get_config_key(self) -> str:
        return "psoma"

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
        params = PSOMAParams(
            pop_size=int(values.get("pop_size", 20)),
            omega=float(values.get("omega", 0.4)),
            c1=float(values.get("c1", 1.5)),
            c2=float(values.get("c2", 2.0)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_freq=int(values.get("local_search_freq", 10)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        solver = PSOMAsSolver(
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
