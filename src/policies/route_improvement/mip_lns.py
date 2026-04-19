"""
MIP Large Neighborhood Search (Exact Ruin and Recreate).

A matheuristic that disrupts the current routing architecture using a spatial
ruin operator, but utilizes an exact Gurobi MILP formulation to re-insert the
fragmented nodes optimally, bridging the speed of LNS with exact bounding.
"""

import logging
import random
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.destroy_ruin.random import random_removal
from logic.src.policies.helpers.operators.intensification_fixing import fix_and_optimize

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy, tour_distance

logger = logging.getLogger(__name__)


@RouteImproverRegistry.register("mip_lns")
class MIPLNSRouteImprover(IRouteImprovement):
    """
    Exact Repair Large Neighborhood Search.

    1. Destroys Q nodes randomly or spatially.
    2. Instead of greedy repair, delegates to the exact solver (Gurobi)
       by isolating the destroyed components and solving the Restricted Sub-MIP.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        dm = to_numpy(kwargs.get("distance_matrix", kwargs.get("distancesC")))
        if dm is None or not tour:
            return tour, {"algorithm": "MIPLNSRouteImprover"}

        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        iterations = kwargs.get("iterations", self.config.get("iterations", 5))
        ruin_pct = kwargs.get("ruin_fraction", self.config.get("ruin_fraction", 0.2))
        seed = kwargs.get("seed", self.config.get("seed", 42))
        rng = random.Random(seed)

        current_tour = [n for n in tour]
        best_tour = [n for n in tour]
        best_cost = tour_distance(split_tour(current_tour), dm)

        try:
            for it in range(iterations):
                routes = split_tour(current_tour)
                total_nodes = sum(len(r) for r in routes)
                q_nodes = max(1, int(total_nodes * ruin_pct))

                # 1. Destroy: Strip nodes from the routes
                partial_routes, removed_nodes = random_removal(routes, q_nodes, rng=rng)

                # We format the problem to use `fix_and_optimize` logic internally.
                # Since we removed nodes, we create a "free" route containing the orphans,
                # forcing the exact solver to distribute them into the active routes.
                orphan_route = [node for node in removed_nodes]
                sub_mip_routes = partial_routes + [orphan_route]

                # 2. Exact Repair: Pass to exact MILP (using fix_and_optimize engine)
                # By locking the completely untouched routes, we force the MILP to solve
                # the re-insertion exactly across the affected subset.
                refined_routes = fix_and_optimize(
                    routes=sub_mip_routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    n_free=len(sub_mip_routes),  # Treat all affected routes as free variables
                    free_fraction=1.0,
                    time_limit=30.0,  # Bounded exact solve
                    seed=seed + it,
                )

                candidate_tour = assemble_tour(refined_routes)
                candidate_cost = tour_distance(refined_routes, dm)

                if candidate_cost < best_cost - 1e-6:
                    best_cost = candidate_cost
                    best_tour = candidate_tour
                    current_tour = candidate_tour

            return best_tour, {"algorithm": "MIPLNSRouteImprover", "iterations": iterations, "final_cost": best_cost}

        except Exception as e:
            logger.warning(f"MIP-LNS failed, falling back to original tour: {e}")
            return tour, {"algorithm": "MIPLNSRouteImprover"}
