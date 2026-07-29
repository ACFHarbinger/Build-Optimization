"""
Held-Karp DP Route Reoptimization Route Improver.

Delegates to operators.intensification.dp_route_reopt (or its profit
variant when revenue/cost are configured) to perform exact TSP
reoptimization per route using Dynamic Programming.
"""

from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.intensification_fixing import (
    dp_route_reopt,
    dp_route_reopt_profit,
)

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("dp_route_reopt")
class DPRouteReoptRouteImprover(IRouteImprovement):
    """
    Held-Karp exact TSP route improver. Reoptimizes each route
    independently using DP. Routes longer than `dp_max_nodes`
    are returned unchanged.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "DPRouteReoptRouteImprover"}

        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        dp_max_nodes = kwargs.get("dp_max_nodes", self.config.get("dp_max_nodes", 20))

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "DPRouteReoptRouteImprover"}

            # DP is exact per-route; additional passes yield no improvement.
            # We deliberately do not expose max_iter as a kwarg.
            if revenue_kg > 0 or cost_per_km > 0:
                refined = dp_route_reopt_profit(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    max_nodes=dp_max_nodes,
                )
            else:
                refined = dp_route_reopt(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    max_nodes=dp_max_nodes,
                )

            return assemble_tour(refined), {"algorithm": "DPRouteReoptRouteImprover"}

        except Exception:
            return tour, {"algorithm": "DPRouteReoptRouteImprover"}
