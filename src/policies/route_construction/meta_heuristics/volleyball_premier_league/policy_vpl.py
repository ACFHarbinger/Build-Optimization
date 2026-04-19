"""
VPL Policy Adapter.

Adapts the Volleyball Premier League (VPL) logic to the agnostic interface.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.vpl import VPLConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.volleyball_premier_league.params import VPLParams
from logic.src.policies.route_construction.meta_heuristics.volleyball_premier_league.solver import VPLSolver


@RouteConstructorRegistry.register("vpl")
class VPLPolicy(BaseRoutingPolicy):
    """
    VPL policy class.

    Visits pre-selected 'mandatory' bins using the population-based VPL metaheuristic
    with dual population structure (active and passive teams).
    """

    def __init__(self, config: Optional[Union[VPLConfig, Dict[str, Any]]] = None):
        """Initialize VPL policy with optional config.

        Args:
            config: VPLConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return VPLConfig

    def _get_config_key(self) -> str:
        """Return config key for VPL."""
        return "vpl"

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
        Execute the Volleyball Premier League (VPL) solver logic.

        VPL is a bio-inspired competition metaheuristic that mimics the
        interaction of teams in a sports league. In this implementation:
        - Teams: A set of candidate solutions organized into a league.
        - Coaching: Solutions interact via peer-to-peer influence (crossover
          and displacement) weighted by coaching parameters.
        - Substitution: Low-performing components (nodes or sub-tours) are
          periodically replaced with better-performing ones to maintain
          solution quality.
        The algorithm uses a dual population structure (active league and
        passive reserves) to balance intensive competition with structural
        diversity.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                VPL parameters (n_teams, substitution_rate, coaching_weights).
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
        params = VPLParams(
            n_teams=values.get("n_teams", 30),
            max_iterations=values.get("max_iterations", 200),
            substitution_rate=values.get("substitution_rate", 0.2),
            coaching_weight_1=values.get("coaching_weight_1", 0.5),
            coaching_weight_2=values.get("coaching_weight_2", 0.3),
            coaching_weight_3=values.get("coaching_weight_3", 0.2),
            elite_size=values.get("elite_size", 3),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 300.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = VPLSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
