"""
Simulated Annealing Policy Adapter.

Wraps the core SA meta-heuristic into the standard BaseRoutingPolicy interface
to permit unified execution by the factory manager.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import SAParams
from .solver import SASolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("sa")
class SAPolicy(BaseRoutingPolicy):
    r"""
    Simulated Annealing (SA) Policy - Stochastic Trajectory-Based Optimization.

    This policy implements the Simulated Annealing metaheuristic, inspired by the
    metallurgical process of heating and controlled cooling to achieve a low-energy
    state.

    Search Logic:
    1.  **Neighbor Generation**: Proposes small, stochastic modifications to the
        current route (e.g., 2-opt, relocate, swap).
    2.  **Acceptance Criterion**: Employs the Metropolis-Hastings rule. It always
        accepts improving moves and accepts deteriorating moves with a probability
        $P = \exp(-\Delta / T)$, where $\Delta$ is the loss in profit and $T$ is
        the current temperature.
    3.  **Cooling Schedule**: Systematically reduces $T$ over time, transitioning
        the search from exploration (early stage) to intensification (late stage).

    SA is robust against premature convergence and is highly effective at escaping
    the narrow basins of attraction in the routing landscape.

    Registry key: ``"sa"``
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        # Using dictionary configurations internally, avoiding external dataclass dependencies
        return None

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
        Execute the Simulated Annealing (SA) solver logic.

        SA is a probabilistic technique for approximating the global optimum of
        a given function. Specifically, it is a metaheuristic to approximate
        global optimization in a large search space for an optimization problem.
        It is often used when the search space is discrete. In this implementation:
        - Annealing Schedule: The search begins at a high "temperature" (accepting
          worse solutions frequently) and slowly cools down.
        - Acceptance: Transitions to worse solutions are allowed with a probability
          determined by the Boltzmann-Metropolis criterion, allowing the solver
          to escape local optima.
        - Markov Chains: Multiple local search moves are attempted at each
          temperature step.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                SA parameters (initial_temp, cooling_rate, iterations_per_temp).
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
        # 1. Parse parameters
        params = SAParams.from_config(values)

        # 2. Instantiate core solver
        solver = SASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Optimize and return
        best_routes, best_profit, best_cost = solver.solve(initial_solution=None)

        return best_routes, best_profit, best_cost
