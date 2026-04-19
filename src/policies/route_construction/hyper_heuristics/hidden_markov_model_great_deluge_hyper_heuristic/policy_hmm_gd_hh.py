"""
HMM-GD Policy Adapter.

Adapts the HMM + Great Deluge (HMM-GD) hyper-heuristic solver to the
agnostic BaseRoutingPolicy interface.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hmm_gd_hh import HMMGDHHConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params import (
    HMMGDHHParams,
)
from logic.src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver import (
    HMMGDHHSolver,
)


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("hmm_gd_hh")
class HMMGDHHPolicy(BaseRoutingPolicy):
    """
    HMM-GD-HH policy class.

    Visits bins using the online-learning HMM + Great Deluge hyper-heuristic.
    The HMM learns which Low-Level Heuristic to apply based on observed search
    states (improving / stagnating / escaping).  The Great Deluge criterion
    provides acceptance control without temperature parameters.
    """

    def __init__(self, config: Optional[Union[HMMGDHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HMMGDHHConfig

    def _get_config_key(self) -> str:
        return "hmm_gd_hh"

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
        Execute the Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH)
        solver logic.

        This solver combines an online-learning HMM for low-level heuristic
        selection with the Great Deluge acceptance criterion. The HMM models
        the search process as a set of hidden states (improving, stagnant, etc.)
        and learns to transition between heuristics to escape local optima.

        The Great Deluge framework provides a parameter-less acceptance threshold
        that "rains" (decreases or increases) over time to control search intensity.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                hyperparameters like `rain_speed`, `learning_rate`, etc.
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = HMMGDHHParams(
            max_iterations=int(values.get("max_iterations", 500)),
            flood_margin=float(values.get("flood_margin", 0.05)),
            rain_speed=float(values.get("rain_speed", 0.001)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = HMMGDHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        rng = random.Random(params.seed)
        heuristic_routes = build_greedy_routes(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            mandatory_nodes=mandatory_nodes,
            rng=rng,
        )

        routes, profit, cost = solver.solve(initial_routes=heuristic_routes)
        return routes, profit, cost
