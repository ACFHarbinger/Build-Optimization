"""
QDE Policy Adapter.

Adapts the Quantum-Inspired Differential Evolution (QDE) logic to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.qde import QDEConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.quantum_differential_evolution.params import QDEParams
from logic.src.policies.quantum_differential_evolution.solver import QDESolver

from .factory import PolicyRegistry


@PolicyRegistry.register("qde")
class QDEPolicy(BaseRoutingPolicy):
    """
    QDE policy class.

    Visits bins using Quantum-Inspired Differential Evolution.
    """

    def __init__(self, config: Optional[Union[QDEConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return QDEConfig

    def _get_config_key(self) -> str:
        return "qde"

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
        params = QDEParams(
            pop_size=int(values.get("pop_size", 20)),
            F=float(values.get("F", 0.5)),
            CR=float(values.get("CR", 0.7)),
            max_iterations=int(values.get("max_iterations", 200)),
            time_limit=float(values.get("time_limit", 60.0)),
            n_removal=int(values.get("n_removal", 2)),
        )

        solver = QDESolver(
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
