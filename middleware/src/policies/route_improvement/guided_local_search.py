"""
Guided Local Search Route Improver.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy, tour_distance


@RouteImproverRegistry.register("guided_local_search")
class GuidedLocalSearchRouteImprover(IRouteImprovement):
    """
    Guided Local Search (GLS) route improver.
    Augments the distance matrix with penalties on frequently used edges
    to escape local optima.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply Guided Local Search to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'gls_iterations',
                     'gls_inner_iterations', 'gls_lambda_factor', 'gls_base_operator', etc.

        Returns:
            List[int]: Refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "GuidedLocalSearchRouteImprover"}

        # Parameters
        gls_iterations = kwargs.get("gls_iterations", self.config.get("gls_iterations", 20))
        inner_iterations = kwargs.get("gls_inner_iterations", self.config.get("gls_inner_iterations", 50))
        lambda_factor_weight = kwargs.get("gls_lambda_factor", self.config.get("gls_lambda_factor", 0.1))
        penalty_decay = kwargs.get("gls_penalty_decay", self.config.get("gls_penalty_decay", 1.0))
        base_op_name = kwargs.get("gls_base_operator", self.config.get("gls_base_operator", "or_opt"))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        if len(tour) < 3:
            return tour, {"algorithm": "GuidedLocalSearchRouteImprover"}

        try:
            from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager

            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "GuidedLocalSearchRouteImprover"}

            penalty = np.zeros_like(dm)
            lambda_factor = lambda_factor_weight * np.mean(dm)

            best_routes = [r[:] for r in routes]
            best_cost = tour_distance(best_routes, dm)

            current_routes = [r[:] for r in routes]

            for _ in range(gls_iterations):
                # 1. Build augmented matrix
                dm_aug = dm + lambda_factor * penalty

                # 2. Run base operator on augmented matrix
                manager = LocalSearchManager(
                    dist_matrix=dm_aug,
                    wastes=wastes,
                    capacity=capacity,
                    R=R,
                    C=C,
                    improvement_threshold=1e-6,
                    seed=seed,
                )
                manager.set_routes(current_routes)

                # Get operator method
                op_meth = self._get_operator_method(manager, base_op_name)
                if op_meth:
                    for _ in range(inner_iterations):
                        if not op_meth():
                            break

                current_routes = manager.get_routes()
                current_cost = tour_distance(current_routes, dm)

                if current_cost < best_cost - 1e-6:
                    best_cost = current_cost
                    best_routes = [r[:] for r in current_routes]

                # 3. Penalize edges with maximum utility in current tour
                self._update_penalties(current_routes, dm, penalty)

                # 4. Optional penalty decay
                if penalty_decay < 1.0:
                    penalty *= penalty_decay

            return assemble_tour(best_routes), {"algorithm": "GuidedLocalSearchRouteImprover"}

        except Exception:
            return tour, {"algorithm": "GuidedLocalSearchRouteImprover"}

    def _get_operator_method(self, manager: Any, name: str):
        """Map operator name to LocalSearchManager method."""
        name = name.lower()
        if name in ["or_opt", "relocate_chain"]:
            return manager.or_opt
        elif name in ["cross_exchange", "cross"]:
            return manager.cross_exchange_op
        elif name in ["2opt", "two_opt"]:
            return manager.two_opt_intra
        elif name in ["swap"]:
            return manager.swap
        elif name in ["relocate"]:
            return manager.relocate
        elif name in ["3opt", "three_opt"]:
            return manager.three_opt_intra
        return manager.or_opt

    def _update_penalties(self, routes: List[List[int]], dm: np.ndarray, penalty: np.ndarray):
        """Identify maximum utility edge and increment penalty."""
        max_utility = -1.0
        max_edge = None

        for route in routes:
            if not route:
                continue

            # Depot legs are deliberately excluded; iterating over route
            # only covers internal edges (u != 0 and v != 0).

            # Nodes: [0, r0, r1, ..., rk, 0]
            # Internal edges
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                # cost(u, v) / (1 + penalty[u, v])
                utility = float(dm[u, v]) / (1.0 + penalty[u, v])
                if utility > max_utility:
                    max_utility = utility
                    max_edge = (u, v)

        if not max_edge:
            return

        u, v = max_edge
        penalty[u, v] += 1
        penalty[v, u] += 1
