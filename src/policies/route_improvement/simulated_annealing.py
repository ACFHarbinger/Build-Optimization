"""
Simulated Annealing Route Improver.

Delegates move acceptance to the pluggable ``BoltzmannAcceptance`` criterion,
threading per-step ``AcceptanceMetrics`` into the returned ``ImprovementMetrics``.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import AcceptanceMetrics, ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion import (
    BoltzmannAcceptance,
)

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, route_distance, route_load, split_tour, to_numpy, tour_distance


@RouteImproverRegistry.register("simulated_annealing")
class SimulatedAnnealingRouteImprover(IRouteImprovement):
    """
    Simulated Annealing route improver.

    Applies random local moves (swaps, relocations, reversals) and delegates
    move acceptance to a ``BoltzmannAcceptance`` criterion, which returns
    ``AcceptanceMetrics`` on each step.  These are collected in an
    ``acceptance_trace`` and surfaced via the ``ImprovementMetrics`` return.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply Simulated Annealing to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'sa_iterations',
                     'sa_t_init', 'sa_t_min', 'sa_cooling', etc.

        Returns:
            Tuple[List[int], ImprovementMetrics]: (refined_tour, metrics)
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "SimulatedAnnealingRouteImprover"}

        # Parameters
        iterations = kwargs.get("sa_iterations", self.config.get("sa_iterations", 5000))
        t_init = kwargs.get("sa_t_init", self.config.get("sa_t_init", 10.0))
        cooling = kwargs.get("sa_cooling", self.config.get("sa_cooling", 0.999))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))

        dm = to_numpy(distance_matrix)

        if len(tour) < 3:
            return tour, {"algorithm": "SimulatedAnnealingRouteImprover"}

        # Instantiate the acceptance criterion (Q2 delegation)
        criterion = BoltzmannAcceptance(initial_temp=t_init, alpha=cooling, seed=seed)
        acceptance_trace: List[AcceptanceMetrics] = []
        n_local_optima = 0

        try:
            rng = np.random.default_rng(seed)
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "SimulatedAnnealingRouteImprover"}

            current_routes = [r[:] for r in routes]
            current_cost = tour_distance(current_routes, dm)
            criterion.setup(current_cost)

            best_routes = [r[:] for r in current_routes]
            best_cost = current_cost

            for i in range(iterations):
                # 1b. Periodic cost resync to avoid drift
                if i > 0 and i % 100 == 0:
                    current_cost = tour_distance(current_routes, dm)

                # 2. Apply a random move
                new_routes, delta = self._random_move(current_routes, dm, rng, wastes, capacity)

                if new_routes is None:
                    continue

                # 3. Delegate acceptance to the criterion (returns metrics now)
                is_accepted, step_metrics = criterion.accept(
                    current_obj=current_cost,
                    candidate_obj=current_cost + delta,
                )
                criterion.step(
                    current_obj=current_cost,
                    candidate_obj=current_cost + delta,
                    accepted=is_accepted,
                )
                # Cap trace length to avoid unbounded memory in long runs
                if len(acceptance_trace) < 1000:
                    acceptance_trace.append(step_metrics)

                if is_accepted:
                    current_routes = new_routes
                    current_cost += delta
                    if current_cost < best_cost - 1e-6:
                        best_cost = current_cost
                        best_routes = [r[:] for r in current_routes]
                else:
                    n_local_optima += 1

            metrics: ImprovementMetrics = {
                "algorithm": "SimulatedAnnealingRouteImprover",
                "n_iterations": iterations,
                "n_local_optima": n_local_optima,
                "best_delta": best_cost - tour_distance(split_tour(tour), dm),
                "acceptance_trace": acceptance_trace,
            }
            return assemble_tour(best_routes), metrics

        except Exception:
            return tour, {"algorithm": "SimulatedAnnealingRouteImprover"}

    def _random_move(  # noqa: C901
        self,
        routes: List[List[int]],
        dm: np.ndarray,
        rng: np.random.Generator,
        wastes: Dict[int, float],
        capacity: float,
    ) -> Tuple[Any, float]:
        """Pick and apply a random move (intra-route or inter-route)."""
        move_type = rng.choice(["swap", "relocate", "two_opt"])

        # Select one or two random routes
        if not routes:
            return None, 0.0

        r1_idx = rng.integers(len(routes))
        r1 = routes[r1_idx]
        if not r1:
            return None, 0.0

        new_routes = [r[:] for r in routes]

        if move_type == "two_opt":
            # Reverses a segment in r1
            if len(r1) < 2:
                return None, 0.0
            # Must have length >= 2 to be a non-trivial 2-opt reversal
            i, j = sorted(rng.choice(len(r1) + 1, size=2, replace=False))
            if j - i < 2:
                return None, 0.0
            # Skip full-reversal no-op (symmetric matrix simplification)
            if i == 0 and j == len(r1):
                return None, 0.0

            old_cost = route_distance(r1, dm)
            r1_new = r1[:i] + r1[i:j][::-1] + r1[j:]
            new_cost = route_distance(r1_new, dm)
            new_routes[r1_idx] = r1_new
            return new_routes, new_cost - old_cost

        elif move_type == "swap":
            # Swaps two nodes (possibly across different routes)
            r2_idx = rng.integers(len(routes))
            r2 = routes[r2_idx]
            if not r2:
                return None, 0.0

            p1 = rng.integers(len(r1))
            p2 = rng.integers(len(r2))

            if r1_idx == r2_idx and p1 == p2:
                return None, 0.0

            # Capacity check
            w1, w2 = wastes.get(r1[p1], 0.0), wastes.get(r2[p2], 0.0)
            if r1_idx != r2_idx:
                if route_load(r1, wastes) - w1 + w2 > capacity:
                    return None, 0.0
                if route_load(r2, wastes) - w2 + w1 > capacity:
                    return None, 0.0

            if r1_idx == r2_idx:
                old_cost = route_distance(routes[r1_idx], dm)
                new_routes[r1_idx][p1], new_routes[r1_idx][p2] = new_routes[r1_idx][p2], new_routes[r1_idx][p1]
                new_cost = route_distance(new_routes[r1_idx], dm)
            else:
                old_cost = route_distance(routes[r1_idx], dm) + route_distance(routes[r2_idx], dm)
                new_routes[r1_idx][p1], new_routes[r2_idx][p2] = new_routes[r2_idx][p2], new_routes[r1_idx][p1]
                new_cost = route_distance(new_routes[r1_idx], dm) + route_distance(new_routes[r2_idx], dm)

            return new_routes, new_cost - old_cost

        elif move_type == "relocate":
            # Moves a node from r1 to r2
            if len(r1) < 1:
                return None, 0.0
            r2_idx = rng.integers(len(routes))
            new_routes[r2_idx] = routes[r2_idx][:]  # Ensure we have a deep copy for r2

            p1 = rng.integers(len(r1))
            p2 = rng.integers(len(new_routes[r2_idx]) + 1)

            # Capacity check
            w1 = wastes.get(r1[p1], 0.0)
            if r1_idx != r2_idx and route_load(new_routes[r2_idx], wastes) + w1 > capacity:
                return None, 0.0

            if r1_idx == r2_idx:
                # Reject no-ops
                if p1 == p2 or p1 + 1 == p2:
                    return None, 0.0
                old_cost = route_distance(routes[r1_idx], dm)
                node = new_routes[r1_idx].pop(p1)
                # Adjust insertion index to account for the pop
                insert_idx = p2 - 1 if p2 > p1 else p2
                new_routes[r1_idx].insert(insert_idx, node)
                new_cost = route_distance(new_routes[r1_idx], dm)
            else:
                old_cost = route_distance(routes[r1_idx], dm) + route_distance(routes[r2_idx], dm)
                node = new_routes[r1_idx].pop(p1)
                new_routes[r2_idx].insert(p2, node)
                new_cost = route_distance(new_routes[r1_idx], dm) + route_distance(new_routes[r2_idx], dm)

            return new_routes, new_cost - old_cost

        return None, 0.0
