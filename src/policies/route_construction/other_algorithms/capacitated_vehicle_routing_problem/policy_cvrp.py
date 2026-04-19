"""
CVRP Policy module.

Implements a multi-vehicle routing policy (CVRP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import CVRPConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

if TYPE_CHECKING:
    from logic.src.interfaces.context.multi_day_context import MultiDayContext
    from logic.src.interfaces.context.search_context import SearchContext

from .cvrp import find_routes, find_routes_ortools
from .params import CVRPParams


@GlobalRegistry.register(
    PolicyTag.HEURISTIC,
    PolicyTag.CONSTRUCTION,
)
@RouteConstructorRegistry.register("cvrp")
class CVRPPolicy(BaseRoutingPolicy):
    """
    Capacitated Vehicle Routing Policy (CVRP).

    Visits provided 'mandatory' bins using multiple vehicles.
    """

    def __init__(self, config: Optional[Union[CVRPConfig, Dict[str, Any]]] = None):
        """Initialize CVRP policy with optional config.

        Args:
            config: CVRPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return CVRPConfig

    def _get_config_key(self) -> str:
        """Return config key for CVRP."""
        return "cvrp"

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
        Run CVRP solver.

        Note: CVRP uses global indices directly, not the subset mapping.
        """
        # Not used - we override execute() for CVRP
        return [[]], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Capacitated Vehicle Routing Problem (CVRP) solver logic.

        This policy solves the problem of finding optimal routes for a fleet
        of vehicles to deliver or collect waste from a set of bins, subject
        to vehicle capacity constraints and mandatory collection requirements.
        It supports multiple engines, including OR-Tools and custom geometric
        heuristics (Clarke-Wright savings).

        Args:
            **kwargs: Context dictionary containing:
                - mandatory (List[int]): Global indices of bins to be collected.
                - bins (Bins): Container for bin stock and distribution data.
                - distance_matrix (np.ndarray): Global symmetric distance matrix.
                - n_vehicles (int): Number of available vehicles.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                A 5-tuple containing:
                - tour: Optimized collection routes (flat or nested list).
                - cost: Total travel cost calculated based on the routes.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - search_context: Propagated or updated search context.
                - multi_day_context: Propagated multi-day state metadata.
        """
        mandatory = kwargs.get("mandatory", [])
        early_result = self._validate_mandatory(mandatory)
        if early_result is not None:
            # Re-map early result to 5-tuple
            routes, dist = early_result
            return routes, dist, 0.0, kwargs.get("search_context"), kwargs.get("multi_day_context")

        bins = kwargs["bins"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        n_vehicles = kwargs.get("n_vehicles", 1)
        cached = kwargs.get("cached")
        coords = kwargs.get("coords")
        config = kwargs.get("config", {})
        distancesC = kwargs.get("distancesC")
        distance_matrix = kwargs.get("distance_matrix", distancesC)

        to_collect = list(mandatory) if mandatory else list(range(1, bins.n + 1))

        # Load capacity and other area-specific constant params
        capacity, _, _, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)

        # Initialize type-safe Params
        params = CVRPParams.from_config(self._config or values)

        # Use cached route if available and no specific mandatory
        if cached is not None and len(cached) > 1 and not mandatory:
            tour = cached
        else:
            seed = kwargs.get("seed") if kwargs.get("seed") is not None else params.seed
            solver_fn = find_routes_ortools if params.engine == "ortools" else find_routes
            tour = solver_fn(
                distancesC,
                bins.c,
                capacity,
                np.array(to_collect),
                n_vehicles,
                coords,
                time_limit=params.time_limit,
                seed=seed,
            )
        # Ensure list format
        if hasattr(tour, "tolist"):
            tour = tour.tolist()
        elif not isinstance(tour, list):
            tour = list(tour)

        cost = self._compute_cost(distance_matrix, tour)
        # Compute profit
        R = values.get("revenue_kg", 1.0)
        C = values.get("cost_km", 1.0)
        visited = {n for n in tour if n != 0}
        collected_revenue = sum(float(bins.c[n - 1]) * R for n in visited if 1 <= n <= bins.n)
        profit = collected_revenue - cost * C

        return tour, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")
