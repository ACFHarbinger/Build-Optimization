"""
Path Refinement Route Improver.
"""

from typing import Any, List, Tuple

from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics

from .base import RouteImproverRegistry


@RouteImproverRegistry.register("path")
class PathRouteImprover(IRouteImprovement):
    """
    Refines the tour by including nodes that lie on the shortest paths between consecutive
    stops in the tour, provided they fit within the vehicle capacity.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Refine the tour by picking up convenient bins along the path.

        Args:
            tour: The current tour (list of bin IDs).
            **kwargs: Context containing 'bins' or 'total_fill', 'paths_between_states',
                      and 'vehicle_capacity'.

        Returns:
            List[int]: The expanded tour including opportunistic pickups.
        """
        bins = kwargs.get("bins")
        paths = kwargs.get("paths_between_states")

        current_fill = kwargs.get("total_fill")
        if current_fill is None and bins is not None:
            current_fill = getattr(bins, "c", None)

        if current_fill is None or paths is None:
            return tour, {"algorithm": "PathRouteImprover"}

        capacity = kwargs.get("max_capacity") or kwargs.get(
            "vehicle_capacity", self.config.get("vehicle_capacity", 100.0)
        )

        selected_nodes = set(tour)
        if 0 in selected_nodes:
            selected_nodes.remove(0)

        current_load = sum(current_fill[node - 1] for node in selected_nodes)

        new_tour = [tour[0]]

        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i + 1]

            try:
                segment = paths[u][v]
            except (IndexError, KeyError):
                segment = [u, v]

            if not segment:
                new_tour.append(v)
                continue

            for node in segment[1:]:
                if node in (0, v):
                    new_tour.append(node)
                    continue

                if node not in selected_nodes:
                    waste = current_fill[node - 1]
                    if current_load + waste <= capacity:
                        current_load += waste
                        selected_nodes.add(node)
                        new_tour.append(node)

        return new_tour, {"algorithm": "PathRouteImprover"}
