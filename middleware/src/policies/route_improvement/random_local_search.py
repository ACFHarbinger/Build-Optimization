"""
Random Local Search Route Improver.
"""

from typing import Any, Callable, List, Tuple

import numpy as np

from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("random_local_search")
class RandomLocalSearchRouteImprover(IRouteImprovement):
    """
    Performs stochastic local search refinement by applying random operators from
    the metaheuristic sub-package. Compatible with multi-route tours.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply random local search operators stochastically.

        Args:
            tour: The initial tour to refine (includes depot 0s).
            **kwargs: Context containing 'distance_matrix', 'iterations', 'params', 'seed'.

        Returns:
            List[int]: The refined tour.
        """

        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or len(tour) < 3:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        # Get parameters from config (via kwargs) with fallbacks
        n_iterations = kwargs.get("iterations", kwargs.get("n_iterations", self.config.get("iterations", 500)))
        op_probs = kwargs.get(
            "params",
            kwargs.get(
                "op_probs",
                self.config.get(
                    "params",
                    {
                        "two_opt": 0.2,
                        "two_opt_star": 0.2,
                        "swap": 0.15,
                        "swap_star": 0.15,
                        "relocate": 0.15,
                        "three_opt": 0.1,
                        "or_opt": 0.05,
                    },
                ),
            ),
        )

        seed = kwargs.get("seed", self.config.get("seed", 42))
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        dist_matrix = to_numpy(distance_matrix)

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        # Initialize LocalSearchManager
        manager = LocalSearchManager(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            improvement_threshold=1e-6,
            seed=seed,
        )
        manager.set_routes(routes)

        # Map probabilistic keys to manager methods
        op_map: dict[str, Callable[[], Any]] = {
            "two_opt": manager.two_opt_intra,
            "swap": manager.swap,
            "relocate": manager.relocate,
            "two_opt_star": manager.two_opt_star,
            "swap_star": manager.swap_star,
            "three_opt": manager.three_opt_intra,
            "or_opt": lambda: manager.or_opt(chain_len=2),
            "relocate_chain": lambda: manager.relocate_chain_op(max_chain_len=3),
            "cross_exchange": manager.cross_exchange_op,
        }

        active_ops = [op for op in op_probs.keys() if op in op_map]
        if not active_ops:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}

        probs = np.array([op_probs.get(op, 0.0) for op in active_ops])
        probs = probs / (probs.sum() + 1e-10)

        rng = np.random.default_rng(seed)

        try:
            for _ in range(n_iterations):
                op_name = rng.choice(active_ops, p=probs)
                op_func = op_map[op_name]
                op_func()  # Apply the operator (manager handles success internally)

            # Re-assemble tour
            return assemble_tour(manager.get_routes()), {"algorithm": "RandomLocalSearchRouteImprover"}
        except Exception:
            return tour, {"algorithm": "RandomLocalSearchRouteImprover"}
