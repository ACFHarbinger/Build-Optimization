"""
TS (Tabu Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ts import TSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.tabu_search.params import TSParams
from logic.src.policies.route_construction.meta_heuristics.tabu_search.solver import TSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ts")
class TSPolicy(BaseRoutingPolicy):
    """
    Tabu Search (TS) Policy - Advanced Local Search with Deterministic Memory.

    Tabu Search enhances local search by using a short-term memory (the "Tabu List")
    to prevent the algorithm from returning to recently visited solutions.

    Key Mechanisms:
    1.  **Tabu List**: Stores forbidden moves (e.g., recently moved nodes) to
        enforce a diversifying search trajectory and cycle avoidance.
    2.  **Aspiration Criterion**: Allows a tabu move if it results in a new
        best-found solution, overriding the tabu status to ensure optimality.
    3.  **Intensification & Diversification**: Periodically restarts or adjusts
        parameters to focus on high-quality regions or explore untouched areas
        of the search space.

    By systematically managing its search history, Tabu Search can navigate
    complex routing topologies with high efficiency and precision.

    Registry key: ``"ts"``
    """

    def __init__(self, config: Optional[Union[TSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return TSConfig

    def _get_config_key(self) -> str:
        return "ts"

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
        Execute the Tabu Search (TS) metaheuristic solver logic.

        TS is a local search-based metaheuristic that uses a flexible memory system
        to prevent the search from returning to recently visited solutions (tabu
        list). This implementation features:
        - Short-term Memory: Prevents cycling using a tabu tenure.
        - Long-term Memory: Frequency-based diversification to explore new
          regions and intensification to refine the best known neighborhoods.
        - Aspiration Criteria: Allows tabu moves if they result in a new global
          best solution.
        - Candidate Lists: Restricts the move evaluations to high-potential
          subsets to improve computational efficiency.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                TS parameters (tabu_tenure, max_iterations, elite_size).
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
        params = TSParams(
            # Short-term memory
            tabu_tenure=int(values.get("tabu_tenure", 7)),
            dynamic_tenure=bool(values.get("dynamic_tenure", True)),
            min_tenure=int(values.get("min_tenure", 5)),
            max_tenure=int(values.get("max_tenure", 15)),
            # Aspiration criteria
            aspiration_enabled=bool(values.get("aspiration_enabled", True)),
            # Long-term memory
            intensification_enabled=bool(values.get("intensification_enabled", True)),
            diversification_enabled=bool(values.get("diversification_enabled", True)),
            intensification_interval=int(values.get("intensification_interval", 100)),
            diversification_interval=int(values.get("diversification_interval", 200)),
            elite_size=int(values.get("elite_size", 5)),
            frequency_penalty_weight=float(values.get("frequency_penalty_weight", 0.1)),
            # Candidate list
            candidate_list_enabled=bool(values.get("candidate_list_enabled", True)),
            candidate_list_size=int(values.get("candidate_list_size", 20)),
            # Strategic oscillation
            oscillation_enabled=bool(values.get("oscillation_enabled", False)),
            feasibility_tolerance=float(values.get("feasibility_tolerance", 0.1)),
            # General search
            max_iterations=int(values.get("max_iterations", 5000)),
            max_iterations_no_improve=int(values.get("max_iterations_no_improve", 500)),
            n_removal=int(values.get("n_removal", 3)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
            # Neighborhood structure
            use_swap=bool(values.get("use_swap", True)),
            use_relocate=bool(values.get("use_relocate", True)),
            use_2opt=bool(values.get("use_2opt", True)),
            use_insertion=bool(values.get("use_insertion", True)),
        )

        solver = TSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
