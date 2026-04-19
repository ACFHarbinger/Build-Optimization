"""
Ruin and Recreate Route Improver.

Delegates Simulated Annealing acceptance to the pluggable
``BoltzmannAcceptance`` criterion (Q2 decision), threading per-step
``AcceptanceMetrics`` into the returned ``ImprovementMetrics``.
"""

import random
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import AcceptanceMetrics, ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.search_heuristics.large_neighborhood_search import (
    apply_lns,
)

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
    upgrade_repair_op_to_profit,
)


@RouteImproverRegistry.register("ruin_recreate")
class RuinRecreateRouteImprover(IRouteImprovement):
    """
    Ruin and Recreate route improver (Large Neighborhood Search).

    Randomly removes a subset of bins and re-inserts them greedily.
    When the ``sa`` acceptance mode is selected, delegates to a
    ``BoltzmannAcceptance`` criterion rather than inlining Metropolis logic.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply Ruin and Recreate to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'lns_iterations',
                     'ruin_fraction', 'lns_acceptance', 'destroy_op', 'repair_op', etc.

        Returns:
            Tuple[List[int], ImprovementMetrics]: (refined_tour, metrics)
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "RuinRecreateRouteImprover"}

        # Parameters
        iterations = (
            kwargs.get("lns_iterations", 0)
            if kwargs.get("lns_iterations") is not None
            else self.config.get("lns_iterations", 100)
        )
        ruin_fraction = (
            kwargs.get("ruin_fraction", 0.0)
            if kwargs.get("ruin_fraction") is not None
            else self.config.get("ruin_fraction", 0.2)
        )
        acceptance = (
            kwargs.get("lns_acceptance", "")
            if kwargs.get("lns_acceptance") is not None
            else self.config.get("lns_acceptance", "best")
        ).lower()
        destroy_op = (
            kwargs.get("destroy_op", "")
            if kwargs.get("destroy_op") is not None
            else self.config.get("destroy_op", "random")
        )
        repair_op_raw = (
            kwargs.get("repair_op", "")
            if kwargs.get("repair_op") is not None
            else self.config.get("repair_op", "greedy")
        )
        repair_k = kwargs.get("repair_k", 0) if kwargs.get("repair_k") is not None else self.config.get("repair_k", 2)

        # Operator-specific parameters
        p = kwargs.get("p", 0.0) if kwargs.get("p") is not None else self.config.get("p", 1.0)
        noise = kwargs.get("noise", 0.0) if kwargs.get("noise") is not None else self.config.get("noise", 0.0)
        if acceptance not in ("best", "sa"):
            raise ValueError(f"Unknown acceptance rule: {acceptance!r}")

        temp = (
            kwargs.get("lns_sa_temperature", 1.0)
            if kwargs.get("lns_sa_temperature") is not None
            else self.config.get("lns_sa_temperature", 1.0)
        )
        seed = kwargs.get("seed", 0) if kwargs.get("seed") is not None else self.config.get("seed", 42)

        # Problem parameters
        wastes = kwargs.get("wastes", {}) if kwargs.get("wastes") is not None else self.config.get("wastes", {})
        capacity = (
            kwargs.get("capacity", float("inf"))
            if kwargs.get("capacity") is not None
            else self.config.get("capacity", float("inf"))
        )
        cost_per_km = (
            kwargs.get("cost_per_km", 0.0)
            if kwargs.get("cost_per_km") is not None
            else self.config.get("cost_per_km", 0.0)
        )
        revenue_kg = (
            kwargs.get("revenue_kg", 0.0)
            if kwargs.get("revenue_kg") is not None
            else self.config.get("revenue_kg", 0.0)
        )

        # Mandatory nodes resolution
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        # Profit-aware upgrade
        repair_op = upgrade_repair_op_to_profit(repair_op_raw, revenue_kg, cost_per_km)

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        # Set up SA criterion if needed (Q2 delegation)
        sa_criterion = None
        if acceptance == "sa":
            from logic.src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion import (
                BoltzmannAcceptance,
            )

            # Use a near-constant alpha since LNS typically runs fewer iterations
            sa_criterion = BoltzmannAcceptance(initial_temp=temp, alpha=0.99, seed=seed + 1)

        acceptance_trace: List[AcceptanceMetrics] = []
        n_local_optima = 0

        try:
            # We split the RNG streams for independence
            lns_rng = random.Random(seed)

            current_routes = split_tour(tour)
            if not current_routes:
                return tour, {"algorithm": "RuinRecreateRouteImprover"}

            current_cost = tour_distance(current_routes, dm)

            if sa_criterion is not None:
                sa_criterion.setup(current_cost)

            best_routes = [r[:] for r in current_routes]
            best_cost = current_cost

            # LNS Loop
            for _ in range(iterations):
                # 1 & 2. Ruin and Recreate via apply_lns
                trial_routes = apply_lns(
                    routes=current_routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    rng=lns_rng,
                    ruin_fraction=ruin_fraction,
                    destroy_op=destroy_op,
                    repair_op=repair_op,
                    repair_k=repair_k,
                    mandatory_nodes=mandatory_nodes,
                    p=p,
                    noise=noise,
                )

                # 3. Acceptance — delegate to criterion when in SA mode
                trial_cost = tour_distance(trial_routes, dm)

                if acceptance == "best":
                    accepted = trial_cost < current_cost - 1e-6
                    step_metrics: AcceptanceMetrics = {
                        "criterion": "best",
                        "accepted": accepted,
                        "delta": trial_cost - current_cost,
                    }
                else:
                    # SA mode: delegate to BoltzmannAcceptance
                    assert sa_criterion is not None
                    accepted, step_metrics = sa_criterion.accept(
                        current_obj=current_cost,
                        candidate_obj=trial_cost,
                    )
                    sa_criterion.step(
                        current_obj=current_cost,
                        candidate_obj=trial_cost,
                        accepted=accepted,
                    )

                if len(acceptance_trace) < 500:
                    acceptance_trace.append(step_metrics)

                if accepted:
                    current_routes = trial_routes
                    current_cost = trial_cost
                    if current_cost < best_cost - 1e-6:
                        best_cost = current_cost
                        best_routes = [r[:] for r in current_routes]
                else:
                    n_local_optima += 1

            metrics: ImprovementMetrics = {
                "algorithm": "RuinRecreateRouteImprover",
                "n_iterations": iterations,
                "n_local_optima": n_local_optima,
                "best_delta": best_cost - tour_distance(split_tour(tour), dm),
                "acceptance_trace": acceptance_trace,
            }
            return assemble_tour(best_routes), metrics

        except Exception:
            return tour, {"algorithm": "RuinRecreateRouteImprover"}
