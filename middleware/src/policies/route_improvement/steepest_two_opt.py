"""
Steepest 2-opt Route Improver.

Delegates to operators.improvement_descent.two_opt_steepest (or its profit
variant when revenue/cost are configured) to drive each route to a strict
intra-route 2-opt local minimum.
"""

from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.improvement_descent import (
    two_opt_steepest,
    two_opt_steepest_profit,
)

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("steepest_two_opt")
class SteepestTwoOptRouteImprover(IRouteImprovement):
    """
    Steepest-descent 2-opt route improver. Drives each route to a strict
    intra-route 2-opt local minimum without changing route membership.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "SteepestTwoOptRouteImprover"}

        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        max_iter = kwargs.get("max_iter", self.config.get("max_iter", 500))

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "SteepestTwoOptRouteImprover"}

            if revenue_kg > 0 or cost_per_km > 0:
                refined = two_opt_steepest_profit(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    max_iter=max_iter,
                )
            else:
                refined = two_opt_steepest(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    max_iter=max_iter,
                )

            return assemble_tour(refined), {"algorithm": "SteepestTwoOptRouteImprover"}

        except Exception:
            return tour, {"algorithm": "SteepestTwoOptRouteImprover"}
