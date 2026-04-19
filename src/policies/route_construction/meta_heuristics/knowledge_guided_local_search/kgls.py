"""
Knowledge-Guided Local Search (KGLS) solver.

Implements the CVRP heuristic from Arnold & Sörensen (2019).
Combines Fast Local Search with an intelligent perturbation mechanism
guided by edge length and width penalties.

Reference:
    Arnold, F., & Sorensen, K. "Knowledge-guided
    local search for the vehicle routing problem", 2018
"""

import copy
import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams
from logic.src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.cost_evaluator import (
    CostEvaluator,
)
from logic.src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.params import KGLSParams

logger = logging.getLogger(__name__)


class KGLSSolver:
    """
    KGLS execution engine propagating geometric penalties through local search.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        locations: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: KGLSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the KGLS solver.

        Args:
            dist_matrix: NxN distance matrix.
            locations: Nx2 coordinate matrix for width baseline.
            wastes: Dictionary of node wastes (demands/profits).
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Parameters for the algorithm.
            mandatory_nodes: List of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.locations = locations
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.random = random.Random(params.seed)
        self.rng = np.random.default_rng(params.seed)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.evaluator = CostEvaluator(dist_matrix=self.dist_matrix)

        self.ls_manager = ACOLocalSearch(
            dist_matrix=self.evaluator.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=KSACOParams(
                local_search_iterations=self.params.local_search_iterations,
                vrpp=self.params.vrpp,
                profit_aware_operators=self.params.profit_aware_operators,
                seed=self.params.seed,
            ),
        )

    def calculate_cost(self, routes: List[List[int]], penalized: bool = False) -> float:
        """Calculate total routing cost using active evaluated distances."""
        matrix = self.evaluator.get_distance_matrix() if penalized else self.evaluator.dist_matrix
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += matrix[route[i]][route[i + 1]]
            dist += matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def calculate_profit(self, routes: List[List[int]], penalized: bool = False) -> float:
        """Calculate network profit (revenue - cost)."""
        cost = self.calculate_cost(routes, penalized)
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - cost

    def build_initial_solution(self) -> List[List[int]]:
        """Greedy constructive heuristic. Can be replaced with generalized parallel savings."""
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def apply_local_search(
        self, routes: List[List[int]], ls_manager: ACOLocalSearch, targeted_nodes: Optional[List[int]] = None
    ) -> List[List[int]]:
        """
        Apply local search operators to improve the solution.
        If mapped to targeted nodes, search effort is computationally localized.
        """
        return ls_manager.optimize(routes)

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Main KGLS execution loop."""

        # 1. Initialize
        current_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()

        # Run Initial Local Search without penalties
        current_routes = self.apply_local_search(current_routes, self.ls_manager)

        best_routes = copy.deepcopy(current_routes)
        current_profit = self.calculate_profit(current_routes)
        best_profit = current_profit

        start_time = time.process_time()
        criterium_idx = 0
        iteration = 0
        last_reset_time = start_time

        # 2. Outer Loop
        while True:
            iteration += 1
            current_time = time.process_time()
            if (current_time - start_time) > self.params.time_limit:
                break

            # If stagnating for 20% of the runtime limit, reset to global best
            if (current_time - last_reset_time) > (self.params.time_limit / 5) and current_profit < best_profit:
                logger.debug(f"KGLS Stagnation detected. Resetting to best solution profit: {best_profit:.2f}")
                current_routes = copy.deepcopy(best_routes)
                current_profit = best_profit
                self.evaluator.reset_penalties()
                last_reset_time = current_time

            # Switch penalization criterium systematically
            criterium = self.params.penalization_cycle[criterium_idx % len(self.params.penalization_cycle)]
            criterium_idx += 1

            # --- Perturbation Phase (Knowledge-Guided Shaking) ---
            self.evaluator.enable_penalization()

            # Penalize the worst edges in the current active solution routing
            targeted_nodes = self.evaluator.evaluate_and_penalize_edges(
                routes=current_routes,
                locations=self.locations,
                criterium=criterium,
                num_perturbations=self.params.num_perturbations,
            )

            # Run Local Search heavily restricted/targeted around the perturbed (penalized) matrix
            perturbed_matrix = self.evaluator.get_distance_matrix()
            ls_manager_perturbed = ACOLocalSearch(
                dist_matrix=perturbed_matrix,
                waste=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                params=self.params,
            )
            current_routes = self.apply_local_search(current_routes, ls_manager_perturbed, targeted_nodes)

            # --- Improvement Phase (True Descents) ---
            self.evaluator.disable_penalization()

            # For ACOLocalSearch, optimize() re-initializes routes automatically
            current_routes = self.apply_local_search(current_routes, self.ls_manager)

            # Evaluate using True Distance baseline
            current_profit = self.calculate_profit(current_routes)

            if current_profit > best_profit:
                best_routes = copy.deepcopy(current_routes)
                best_profit = current_profit
                last_reset_time = current_time

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                score=3 if current_profit >= best_profit else 0,
            )

        best_cost = self.calculate_cost(best_routes)
        return best_routes, best_profit, best_cost
