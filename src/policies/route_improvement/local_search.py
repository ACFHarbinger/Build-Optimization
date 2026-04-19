"""
Classical Local Search Route Improver.
"""

from typing import Any, List, Tuple

from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.operators.intensification_fixing import INTENSIFICATION_OPERATORS

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@RouteImproverRegistry.register("classical_local_search")
class ClassicalLocalSearchRouteImprover(IRouteImprovement):
    """
    Wrapper for classical local search operators from the metaheuristic sub-package.
    Drives a multi-route tour to a local minimum using steepest descent or
    iterative improvement.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply classical local search to the tour.

        Args:
            tour: The initial tour to refine (includes depot 0s).
            **kwargs: Context containing 'distance_matrix', 'iterations', 'ls_operator',
                     'wastes', 'capacity', 'R', 'C'.

        Returns:
            List[int]: The refined tour after applying the local search operator.
        """

        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Get parameters from config (via kwargs) with fallbacks
        max_iter = kwargs.get("iterations", kwargs.get("n_iterations", self.config.get("iterations", 500)))
        operator_name = kwargs.get("ls_operator", kwargs.get("operator_name", self.config.get("ls_operator", "2opt")))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        dist_matrix = to_numpy(distance_matrix)

        if len(tour) < 3:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Case 1: Steepest-descent intensification operators
        Mapping = {
            "2opt": "2OPT_PROFIT",
            "two_opt": "2OPT_PROFIT",
            "swap": "NODE_SWAP_PROFIT",
            "relocate": "OR_OPT_PROFIT",
            "or_opt": "OR_OPT_PROFIT",
            "dp_reopt": "DP_REOPT_PROFIT",
            "fix_opt": "FIX_OPT_PROFIT",
            "sp_polish": "SP_POLISH_PROFIT",
        }

        op_key = Mapping.get(operator_name.lower())
        if op_key and op_key in INTENSIFICATION_OPERATORS:
            op_fn = INTENSIFICATION_OPERATORS[op_key]
            try:
                refined_routes = op_fn(routes, dist_matrix, wastes, capacity, R=R, C=C, max_iter=max_iter)  # type: ignore[operator]
                return assemble_tour(refined_routes), {"algorithm": "ClassicalLocalSearchRouteImprover"}
            except Exception:
                return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Case 2: Multi-move or iterative operators via LocalSearchManager
        from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager

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

        # Mapping for Manager methods
        Manager_Mapping = {
            "3opt": manager.three_opt_intra,
            "three_opt": manager.three_opt_intra,
            "2opt*": manager.two_opt_star,
            "two_opt_star": manager.two_opt_star,
            "swap_star": manager.swap_star,
            "4opt": manager.four_opt_intra,
            "four_opt": manager.four_opt_intra,
        }

        op_meth = Manager_Mapping.get(operator_name.lower())
        if op_meth:
            try:
                for _ in range(max_iter):
                    if not op_meth():
                        break
                return assemble_tour(manager.get_routes()), {"algorithm": "ClassicalLocalSearchRouteImprover"}
            except Exception:
                return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}
