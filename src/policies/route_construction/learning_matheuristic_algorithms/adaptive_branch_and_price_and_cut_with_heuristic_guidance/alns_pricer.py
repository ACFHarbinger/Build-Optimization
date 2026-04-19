"""
ALNS Metaheuristic Pricing logic.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver, Route


class ALNSMultiPeriodPricer:
    """
    ALNS-based column generator that navigates the reduced cost landscape defined
    by scenario-augmented prizes. Acts as a heuristic pricing subproblem solver.
    """

    def __init__(self, exact_pricer: RCSPPSolver, rng_seed: int = 42):
        self.exact_pricer = exact_pricer
        self.rng = np.random.default_rng(rng_seed)

    def scenario_overflow_removal(
        self, route_nodes: List[int], scenario_prizes: Dict[int, float], dual_values: Dict[int, float], num_remove: int
    ) -> List[int]:
        """
        Iteratively remove node i from route k that minimizes (pi_i^scenario - pi_i^dual).
        This destroys routes containing bins that have low overflow urgency across the tree.
        """
        if not route_nodes:
            return []

        # Calculate scores: scenario_prize - dual_value
        scores = []
        for i in route_nodes:
            if i == 0:
                scores.append(float("inf"))  # Don't remove depot
            else:
                score = scenario_prizes.get(i, 0.0) - dual_values.get(i, 0.0)
                scores.append(score)

        # Sort indices by score (ascending)
        sorted_indices = np.argsort(scores)

        # Remove the ones with minimum score
        to_remove = set()
        count = 0
        for idx in sorted_indices:
            if route_nodes[idx] != 0 and count < num_remove:
                to_remove.add(route_nodes[idx])
                count += 1

        new_route = [n for n in route_nodes if n not in to_remove]
        return new_route

    def scenario_aware_insertion(
        self,
        route_nodes: List[int],
        unrouted: List[int],
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        dist_matrix: np.ndarray,
        num_insert: int,
    ) -> List[int]:
        """
        Score unrouted candidate nodes based on multi-period expected value rather than
        static fill. Insert node i that maximizes spatial insertion savings weighted by pi_i^scenario.
        """
        current_route = list(route_nodes)

        for _ in range(num_insert):
            if not unrouted:
                break

            best_node = -1
            best_pos = -1
            best_score = -float("inf")

            for u in unrouted:
                # Calculate marginal prize contribution
                prize = scenario_prizes.get(u, 0.0) - dual_values.get(u, 0.0)

                # Evaluate spatial insertion at all positions
                for i in range(len(current_route) - 1):
                    # Cost = dist(route[i], u) + dist(u, route[i+1]) - dist(route[i], route[i+1])
                    cost = (
                        dist_matrix[current_route[i], u]
                        + dist_matrix[u, current_route[i + 1]]
                        - dist_matrix[current_route[i], current_route[i + 1]]
                    )

                    # Score is weighted savings (prize - cost)
                    # We want to maximize this score
                    score = prize - cost
                    if score > best_score:
                        best_score = score
                        best_node = u
                        best_pos = i + 1

            if best_node != -1 and best_score > 0:  # Only insert if profitable
                current_route.insert(best_pos, best_node)
                unrouted.remove(best_node)
            else:
                break

        return current_route

    def _calculate_reduced_cost(
        self,
        route_nodes: List[int],
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        dist_matrix: np.ndarray,
    ) -> float:
        """
        Evaluates routes based on formula:
        c'_k = -dist(k) + sum_i(pi_i^scenario - pi_i^dual)
        """
        dist_cost = 0.0
        prize_sum = 0.0

        for i in range(len(route_nodes) - 1):
            dist_cost += dist_matrix[route_nodes[i], route_nodes[i + 1]]

        for n in route_nodes:
            if n != 0:
                prize_sum += scenario_prizes.get(n, 0.0) - dual_values.get(n, 0.0)

        return prize_sum - dist_cost

    def solve(
        self,
        dual_values: Dict[int, float],
        scenario_prizes: Dict[int, float],
        dist_matrix: np.ndarray,
        initial_routes: List[List[int]],
        all_nodes: List[int],
        max_routes: int = 5,
        rc_tolerance: float = 1e-4,
        alns_iterations: int = 50,
        **kwargs: Any,
    ) -> Tuple[List[Route], bool]:
        """
        Run ALNS Pricing. If it fails to find strictly profitable columns (>1e-4),
        fallback to exact ESPPRC solver.
        """
        best_routes: List[Route] = []

        for init_route in initial_routes:
            current_route = list(init_route)

            for _ in range(alns_iterations):
                unrouted = [n for n in all_nodes if n not in current_route and n != 0]

                # Apply operators
                num_remove = max(1, len(current_route) // 4)
                current_route = self.scenario_overflow_removal(current_route, scenario_prizes, dual_values, num_remove)

                num_insert = num_remove
                current_route = self.scenario_aware_insertion(
                    current_route, unrouted, scenario_prizes, dual_values, dist_matrix, num_insert
                )

                rc = self._calculate_reduced_cost(current_route, scenario_prizes, dual_values, dist_matrix)

                # VRPP is maximization, so reduced cost > 0 is improving.
                # (Mathematical instruction: strictly negative reduced cost < -10^-4 for minimization)
                if rc > rc_tolerance:
                    r = Route(
                        nodes=[n for n in current_route if n != 0],
                        cost=0.0,
                        revenue=0.0,
                        load=0.0,
                        node_coverage=set(current_route),
                    )
                    r.reduced_cost = rc
                    best_routes.append(r)

        best_routes = sorted(best_routes, key=lambda x: x.reduced_cost if x.reduced_cost else 0, reverse=True)[
            :max_routes
        ]

        if best_routes:
            return best_routes, False

        # Fallback to Exact ESPPRC
        return self.exact_pricer.solve(dual_values=dual_values, max_routes=max_routes, **kwargs), True
