"""
Or-opt Route Improver.
"""

from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("or_opt")
class OrOptRouteImprover(IRouteImprovement):
    """
    Or-opt route improver that relocates chains of nodes within or between routes.
    Wraps LocalSearchManager.or_opt.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply Or-opt improvement to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'chain_len', 'iterations', etc.

        Returns:
            List[int]: Refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "OrOptRouteImprover"}

        # Parameters
        chain_len = kwargs.get("chain_len", self.config.get("chain_len", 2))
        iterations = kwargs.get("iterations", kwargs.get("max_iterations", self.config.get("iterations", 500)))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)
        if len(tour) < 3:
            return tour, {"algorithm": "OrOptRouteImprover"}

        try:
            from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager

            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "OrOptRouteImprover"}

            manager = LocalSearchManager(
                dist_matrix=dm,
                wastes=wastes,
                capacity=capacity,
                R=R,
                C=C,
                improvement_threshold=1e-6,
                seed=seed,
            )
            manager.set_routes(routes)

            for _ in range(iterations):
                if not manager.or_opt(chain_len=chain_len):
                    break

            return assemble_tour(manager.get_routes()), {"algorithm": "OrOptRouteImprover"}

        except Exception:
            # Fallback to original tour on any unexpected error
            return tour, {"algorithm": "OrOptRouteImprover"}
