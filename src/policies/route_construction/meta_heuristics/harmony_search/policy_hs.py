"""
HS Policy Adapter.

Adapts the Harmony Search (HS) solver to the agnostic BaseRoutingPolicy
interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hs import HSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import HSParams
from .solver import HSSolver


@RouteConstructorRegistry.register("hs")
class HSPolicy(BaseRoutingPolicy):
    """
    Harmony Search (HS) Policy - Music-Inspired Global Optimization.

    Harmony Search mimics the improvisation process of musicians to find an
    aesthetically pleasing harmony, which corresponds to the global optimum
    in optimization.

    Algorithm Components:
    1.  **Harmony Memory (HM)**: Stores a set of high-quality routing solutions.
    2.  **Improvisation**: Generates a new harmony by either (a) selecting from
        HM, (b) adjusting a value from HM, or (c) choosing a random value,
        guided by Memory Consideration Rate (HMCR) and Pitch Adjustment Rate (PAR).
    3.  **Update**: Replaces the worst harmony in HM if the new harmony is
        superior.

    HS is known for its simplicity and ability to find near-optimal solutions
    rapidly across various combinatorial structures.

    Registry key: ``"hs"``
    """

    def __init__(self, config: Optional[Union[HSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HSConfig

    def _get_config_key(self) -> str:
        return "hs"

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
        Execute the Harmony Search (HS) metaheuristic solver logic.

        HS is a music-inspired metaheuristic based on the process of searching
        for a perfect state of harmony. In this implementation:
        - Harmony Memory (HM): Stores a population of good solutions.
        - Memory Consideration: New harmonies are constructed by choosing
          components from the HM with probability HMCR.
        - Pitch Adjustment: components chosen from memory are optionally
          modified with probability PAR.
        - Random Selection: Remaining components are chosen randomly to maintain
          diversity.
        This policy applies HS to the VRPP using discrete neighborhood operators
        for pitch adjustment.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                HS parameters (hm_size, HMCR, PAR, BW).
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
        params = HSParams(
            hm_size=int(values.get("hm_size", 10)),
            HMCR=float(values.get("HMCR", 0.9)),
            PAR=float(values.get("PAR", 0.3)),
            BW=float(values.get("BW", 0.05)),
            max_iterations=int(values.get("max_iterations", 500)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = HSSolver(
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
