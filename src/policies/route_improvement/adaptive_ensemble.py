"""
Adaptive Ensemble Route Improver.

A meta-algorithmic orchestrator that dynamically selects and executes high-level
route improvement strategies (phases) using a performance-weighted Roulette Wheel.
"""

import random
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import split_tour, to_numpy, tour_distance


@RouteImproverRegistry.register("adaptive_ensemble")
class AdaptiveEnsembleRouteImprover(IRouteImprovement):
    """
    Dynamically orchestrates multiple route improvement algorithms.
    Maintains a probability distribution over available algorithms and updates
    their selection weights based on real-time objective improvements.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        dm = to_numpy(kwargs.get("distance_matrix", kwargs.get("distancesC")))
        if dm is None or not tour:
            return tour, {"algorithm": "AdaptiveEnsembleRouteImprover"}

        algorithms = kwargs.get(
            "phases", self.config.get("phases", ["steepest_two_opt", "ruin_recreate", "cross_exchange"])
        )
        max_iterations = kwargs.get("iterations", self.config.get("iterations", 50))
        reaction_factor = kwargs.get("reaction_factor", self.config.get("reaction_factor", 0.1))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        rng = random.Random(seed)

        # Initialize uniform weights
        weights = {algo: 1.0 for algo in algorithms}

        current_tour = [n for n in tour]
        current_cost = tour_distance(split_tour(current_tour), dm)
        best_tour = [n for n in tour]
        best_cost = current_cost

        trace = []

        for iteration in range(max_iterations):
            # 1. Roulette Wheel Selection
            total_weight = sum(weights.values())
            probs = [weights[a] / total_weight for a in algorithms]
            selected_algo = rng.choices(algorithms, weights=probs, k=1)[0]

            # 2. Instantiate and Execute
            improver_cls = RouteImproverRegistry.get_route_improver_class(selected_algo)
            if improver_cls is None:
                continue

            improver = improver_cls(config=self.config)

            # Limit inner iterations to prevent deep convergence per step
            inner_kwargs = kwargs.copy()
            inner_kwargs["iterations"] = 10

            candidate_tour, step_metrics = improver.process(current_tour, **inner_kwargs)
            candidate_cost = tour_distance(split_tour(candidate_tour), dm)

            # 3. Evaluate and Assign Credit
            delta = current_cost - candidate_cost  # Positive means improvement
            reward = max(0.0, delta)

            if candidate_cost < best_cost - 1e-6:
                best_cost = candidate_cost
                best_tour = [n for n in candidate_tour]
                reward += 10.0  # Bonus for global best

            if delta > 0:
                current_tour = candidate_tour
                current_cost = candidate_cost

            # 4. Exponential Moving Average Weight Update
            weights[selected_algo] = (1.0 - reaction_factor) * weights[selected_algo] + (reaction_factor * reward)
            # Ensure weights don't collapse to zero
            weights[selected_algo] = max(0.1, weights[selected_algo])

            trace.append(
                {
                    "iteration": iteration,
                    "selected_algorithm": selected_algo,
                    "delta": delta,
                    "weight_distribution": {k: round(v, 3) for k, v in weights.items()},
                }
            )

        metrics: ImprovementMetrics = {
            "algorithm": "AdaptiveEnsembleRouteImprover",
            "iterations": max_iterations,
            "final_weights": weights,
            "trace": trace,
        }

        return best_tour, metrics
