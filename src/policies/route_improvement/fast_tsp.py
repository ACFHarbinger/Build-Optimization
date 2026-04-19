"""
Fast TSP Refinement Route Improver.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import find_route

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("fast_tsp")
class FastTSPRouteImprover(IRouteImprovement):
    """
    Refines all sub-tours using the fast_tsp library.
    Splits long tours by depot (0), re-optimizes each segment, and reconstructs.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Refine the tour by splitting it into trips and optimizing each with fast_tsp.

        Args:
            tour: The initial tour to refine (list of node IDs).
            **kwargs: Keyword arguments containing 'distance_matrix'.

        Returns:
            List[int]: The optimized tour with reduced total distance.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        dm = to_numpy(distance_matrix)

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "FastTSPRouteImprover"}

        refined_routes = []
        for trip in routes:
            if len(trip) > 1:
                # Re-optimize with fast_tsp
                time_limit = kwargs.get("time_limit", self.config.get("time_limit", 2.0))
                seed = kwargs.get("seed", self.config.get("seed", 42))
                refined_trip = find_route(
                    dm,
                    np.array(trip),
                    time_limit=time_limit,
                    seed=seed,
                )
                # strip depot 0s from the constructed route to avoid doubles
                refined_routes.append([n for n in refined_trip if n != 0])
            else:
                refined_routes.append(trip)

        return assemble_tour(refined_routes), {"algorithm": "FastTSPRouteImprover"}
