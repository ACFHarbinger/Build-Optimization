import pickle
import random
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.search_heuristics.large_neighborhood_search import (
    apply_lns,
)

from .base import RouteImproverRegistry
from .common.bandit import ThompsonBandit
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
    upgrade_repair_op_to_profit,
)


@RouteImproverRegistry.register("adaptive_large_neighborhood_search")
class AdaptiveLargeNeighborhoodSearchRouteImprover(IRouteImprovement):
    """
    Adaptive Large Neighborhood Search (ALNS) route improver.
    Extends LNS with multiple ruin and recreate operators selected via a bandit.
    Delegates to operators.heuristics.apply_lns for the ruin-and-recreate pass.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply ALNS to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'alns_iterations',
                     'ruin_fraction', 'alns_bandit_warm_start_path', etc.

        Returns:
            List[int]: Refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "AdaptiveLargeNeighborhoodSearchRouteImprover"}

        # Parameters
        iterations = kwargs.get("alns_iterations", self.config.get("alns_iterations", 200))
        ruin_fraction = kwargs.get("ruin_fraction", self.config.get("ruin_fraction", 0.2))
        warm_start_path = kwargs.get("alns_bandit_warm_start_path", self.config.get("alns_bandit_warm_start_path"))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Operator parameters
        p = kwargs.get("p", self.config.get("p", 1.0))
        noise = kwargs.get("noise", self.config.get("noise", 0.0))
        repair_k = kwargs.get("repair_k", self.config.get("repair_k", 2))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))

        # Mandatory nodes resolution
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        try:
            # We use a dedicated RNG for operator selection and perturbation
            lns_rng = random.Random(seed)

            current_routes = split_tour(tour)
            if not current_routes:
                return tour, {"algorithm": "AdaptiveLargeNeighborhoodSearchRouteImprover"}

            current_cost = tour_distance(current_routes, dm)

            best_routes = [r[:] for r in current_routes]
            best_cost = current_cost

            # Initialize Bandits with configurable canonical operator names
            ruin_ops = kwargs.get(
                "alns_ruin_ops", self.config.get("alns_ruin_ops", ["random", "worst", "shaw", "cluster"])
            )
            repair_ops_raw = kwargs.get("alns_repair_ops", self.config.get("alns_repair_ops", ["greedy", "regret"]))

            # Profit-aware upgrade for repair candidates
            repair_ops = [upgrade_repair_op_to_profit(op, revenue_kg, cost_per_km) for op in repair_ops_raw]

            ruin_bandit = ThompsonBandit(ruin_ops, seed=seed)
            recreate_bandit = ThompsonBandit(repair_ops, seed=seed + 1)

            if warm_start_path:
                try:
                    with open(warm_start_path, "rb") as f:
                        weights = pickle.load(f)
                        ruin_bandit.load_weights(weights.get("ruin", {}))
                        recreate_bandit.load_weights(weights.get("recreate", {}))
                except (FileNotFoundError, pickle.UnpicklingError, KeyError, PermissionError):
                    pass

            for _ in range(iterations):
                # 1. Selection
                ruin_name = ruin_bandit.select()
                recreate_name = recreate_bandit.select()

                # 2. Ruin & Recreate via apply_lns
                trial_routes = apply_lns(
                    routes=current_routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    rng=lns_rng,
                    ruin_fraction=ruin_fraction,
                    destroy_op=ruin_name,
                    repair_op=recreate_name,
                    repair_k=repair_k,
                    mandatory_nodes=mandatory_nodes,
                    p=p,
                    noise=noise,
                )

                # 3. Success Check & Update
                # TODO: If SA acceptance is added, split RNG streams:
                # lns_rng = random.Random(seed), sa_rng = random.Random(seed + 1)
                trial_cost = tour_distance(trial_routes, dm)
                success = 0
                if trial_cost < best_cost - 1e-6:
                    success = 1
                    best_cost = trial_cost
                    best_routes = [r[:] for r in trial_routes]

                if trial_cost < current_cost - 1e-6:
                    current_routes = trial_routes
                    current_cost = trial_cost

                ruin_bandit.update(ruin_name, success)
                recreate_bandit.update(recreate_name, success)

            return assemble_tour(best_routes), {"algorithm": "AdaptiveLargeNeighborhoodSearchRouteImprover"}

        except Exception:
            return tour, {"algorithm": "AdaptiveLargeNeighborhoodSearchRouteImprover"}
