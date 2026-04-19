"""
LKH (Lin-Kernighan-Helsgaun) Route Improver.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import solve_lkh

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("lkh")
class LinKernighanHelsgaunRouteImprover(IRouteImprovement):
    """
    Refines tours using the Lin-Kernighan-Helsgaun heuristic for TSP.

    Handles VRPP subset routes by extracting a dense sub-matrix containing
    only the visited nodes, running LKH on this sub-problem, and mapping
    the results back to the original node IDs.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply LKH refinement to a tour using sub-matrix extraction.

        For VRPP instances where only a subset of nodes are visited, this
        method creates a dense sub-problem by extracting only the relevant
        rows and columns from the full distance matrix. This prevents index
        out-of-bounds errors and eliminates performance bottlenecks from
        hallucinated nodes during tour merging.

        Args:
            tour: The initial tour to refine (list of node IDs from the full problem).
            **kwargs: Must contain 'distance_matrix'. Optionally 'max_iterations' and 'seed'.

        Returns:
            List[int]: The refined tour with original node IDs.
        """
        # 1. Early Exits & Edge Cases
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour, {"algorithm": "LinKernighanHelsgaunRouteImprover"}

        dm = to_numpy(distance_matrix)

        if not tour:
            return tour, {"algorithm": "LinKernighanHelsgaunRouteImprover"}

        # 2. Split tour into trips (sub-problems)
        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "LinKernighanHelsgaunRouteImprover"}

        refined_routes = []
        for trip in routes:
            if len(trip) < 2:
                refined_routes.append(trip)
                continue

            # 3. Node Mapping for this Trip
            # Ensure depot (0) is included even if not explicitly in the trip list (split_tour excludes 0s)
            unique_nodes = [0] + sorted(list(set(trip)))
            node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

            # 4. Extract Sub-Matrix for this Trip
            sub_matrix = dm[np.ix_(unique_nodes, unique_nodes)]

            # 5. Translate Trip to Dense Indices
            # LKH expects a cycle. For a trip [n1, n2, ..., nk], we form [0, n1, n2, ..., nk, 0]
            # but in dense indices it's [node_to_idx[0], node_to_idx[n1], ...]
            sub_tour_indices = [node_to_idx[0]] + [node_to_idx[node] for node in trip] + [node_to_idx[0]]

            # 6. Execute LKH
            max_iterations = kwargs.get("max_iterations", self.config.get("max_iterations", 1000))
            max_k = kwargs.get("max_k", self.config.get("max_k", 3))
            seed = kwargs.get("seed", self.config.get("seed", 42))

            try:
                optimized_indices, _ = solve_lkh(
                    sub_matrix,
                    initial_tour=sub_tour_indices,
                    max_iterations=max_iterations,
                    max_k=max_k,
                    seed=seed,
                )
                # Map back and strip depot 0s (assemble_tour will re-add them)
                refined_trip = [unique_nodes[idx] for idx in optimized_indices if unique_nodes[idx] != 0]
                refined_routes.append(refined_trip)
            except Exception:
                # If LKH fails for one trip, keep original trip
                refined_routes.append(trip)

        return assemble_tour(refined_routes), {"algorithm": "LinKernighanHelsgaunRouteImprover"}
