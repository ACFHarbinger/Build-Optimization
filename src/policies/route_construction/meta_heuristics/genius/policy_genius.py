"""
GENIUS (GENI + US) Policy Adapter.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.genius import GENIUSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import GENIUSParams
from .solver import GENIUSSolver


@RouteConstructorRegistry.register("genius")
class GENIUSPolicy(BaseRoutingPolicy):
    """GENIUS (GENI + US) policy class."""

    def __init__(self, config: Optional[Union[GENIUSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GENIUSConfig

    def _get_config_key(self) -> str:
        return "genius"

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
        Execute the GENIUS (Generalized Insertion + Unstringing and Stringing)
        solver logic.

        GENIUS is a classical metaheuristic for the TSP and VRP that uses:
        - GENI (Generalized Insertion): A complex insertion procedure that
          considers multiple neighborhood nodes to find the best placement for
          a new node.
        - US (Unstringing and Stringing): A post-optimization procedure that
          removes nodes (unstringing) and re-inserts them (stringing) to
          improve the tour quality.
        This policy adapts the Gendreau et al. (1992) methodology for the
        WSmart-Route framework, supporting profit-aware extensions.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                GENIUS parameters (neighborhood_size, unstring_type, string_type).
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
        params = GENIUSParams(
            neighborhood_size=int(values.get("neighborhood_size", 5)),
            unstring_type=int(values.get("unstring_type", 1)),
            string_type=int(values.get("string_type", 1)),
            n_iterations=int(values.get("n_iterations", 1)),
            random_us_sampling=bool(values.get("random_us_sampling", False)),
            vrpp=bool(values.get("vrpp", False)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
        )

        solver = GENIUSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
