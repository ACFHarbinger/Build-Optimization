"""
QDE Policy Adapter.

Adapts the Quantum-Inspired Differential Evolution (QDE) logic to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.qde import QDEConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import QDEParams
from .solver import QDESolver


@RouteConstructorRegistry.register("qde")
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
        """
        Execute the Quantum-Inspired Differential Evolution (QDE) solver logic.

        QDE is a hybrid metaheuristic that maps the principles of quantum
        computation (qubits and quantum gates) onto the Differential Evolution
        (DE) framework. In this implementation:
        - Quantum Population: Candidates are represented as quantum-state vectors.
        - Differential Evolution operators (mutation, crossover) are applied
          to the qubits' rotation angles.
        - Observation: Quantum states are collapsed ("measured") into discrete
          VRPP solutions periodically.
        This approach leverages the high exploration capability of quantum-state
        superpositions while utilizing DE's robust intensification mechanisms.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                QDE parameters (pop_size, F, CR, delta_theta).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = QDEParams(
            pop_size=int(values.get("pop_size", 20)),
            F=float(values.get("F", 0.5)),
            CR=float(values.get("CR", 0.7)),
            max_iterations=int(values.get("max_iterations", 200)),
            time_limit=float(values.get("time_limit", 60.0)),
            delta_theta=float(values.get("delta_theta", 0.01 * 3.14159)),
            local_search_iterations=int(values.get("local_search_iterations", 100)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
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
