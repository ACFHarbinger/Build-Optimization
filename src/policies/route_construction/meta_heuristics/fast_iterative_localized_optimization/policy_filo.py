"""
FILO Policy Adapter.

Adapts the Fast Iterative Localized Optimization (FILO) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import FILOConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .filo import FILOSolver
from .params import FILOParams


@RouteConstructorRegistry.register("filo")
class FILOPolicy(BaseRoutingPolicy):
    """
    FILO policy class.

    Visits pre-selected 'mandatory' bins using Fast Iterative Localized Optimization.
    """

    def __init__(self, config: Optional[Union[FILOConfig, Dict[str, Any]]] = None):
        """Initialize FILO policy with optional config.

        Args:
            config: FILOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return FILOConfig

    def _get_config_key(self) -> str:
        """Return config key for FILO."""
        return "filo"

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
        Execute the Fast Iterative Localized Optimization (FILO) solver logic.

        FILO is a high-performance metaheuristic designed for large-scale VRPs.
        It employs a localized search strategy, focusing on subsets of nodes
        (localized neighborhoods) to improve the solution iteratively. It
        combines simulated annealing-based acceptance criteria with sophisticated
        "ruin and recreate" operators tailored for waste collection constraints.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                FILO parameters (max_iterations, temperature factors, shaking intensity).
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
        # Build parameters from config
        params = FILOParams(
            time_limit=float(values.get("time_limit", 60.0)),
            max_iterations=int(values.get("max_iterations", 100)),
            initial_temperature_factor=float(values.get("initial_temperature_factor", 10.0)),
            final_temperature_factor=float(values.get("final_temperature_factor", 100.0)),
            shaking_lb_factor=float(values.get("shaking_lb_factor", 0.5)),
            shaking_ub_factor=float(values.get("shaking_ub_factor", 2.0)),
            shaking_lb_intensity=float(values.get("shaking_lb_intensity", 0.1)),
            shaking_ub_intensity=float(values.get("shaking_ub_intensity", 0.5)),
            delta_gamma=float(values.get("delta_gamma", 0.1)),
            gamma_base=float(values.get("gamma_base", 1.0)),
            gamma_lambda=float(values.get("gamma_lambda", 2.0)),
            omega_base_multiplier=float(values.get("omega_base_multiplier", 1.0)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            n_cw=int(values.get("n_cw", 100)),
            svc_size=int(values.get("svc_size", 50)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = FILOSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
