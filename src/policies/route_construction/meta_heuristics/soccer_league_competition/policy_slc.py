"""
SLC Policy Adapter.

Adapts the Soccer League Competition (SLC) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.slc import SLCConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.soccer_league_competition.params import SLCParams
from logic.src.policies.route_construction.meta_heuristics.soccer_league_competition.solver import SLCSolver


@RouteConstructorRegistry.register("slc")
class SLCPolicy(BaseRoutingPolicy):
    """
    SLC policy class.

    Visits bins using the Soccer League Competition algorithm.
    """

    def __init__(self, config: Optional[Union[SLCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SLCConfig

    def _get_config_key(self) -> str:
        return "slc"

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
        Execute the Soccer League Competition (SLC) solver logic.

        SLC is a competition-inspired metaheuristic based on the dynamics of
        soccer teams in a league. In this implementation:
        - Teams: A population of solutions organized into "teams".
        - Competitions (Matches): Teams compete, and individuals within teams
          interact (via operators like crossover and mutation) to improve their
          skills (profit).
        - Promotion/Relegation: Poorly performing solutions are relegated
          (replaced with new random configurations) to maintain diversity, while
          top performers are promoted (refined with intensification operators).
        This hierarchy creates a multi-layered search that efficiently balances
        global exploration with peer-influenced intensification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                SLC parameters (n_teams, team_size, stagnation_limit).
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
        params = SLCParams(
            n_teams=int(values.get("n_teams", 5)),
            team_size=int(values.get("team_size", 4)),
            max_iterations=int(values.get("max_iterations", 50)),
            stagnation_limit=int(values.get("stagnation_limit", 5)),
            n_removal=int(values.get("n_removal", 1)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SLCSolver(
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
