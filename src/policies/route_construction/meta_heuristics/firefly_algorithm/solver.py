"""
Discrete Firefly Algorithm (DFA) for VRPP.

Adapts the continuous FA to discrete routing by replacing Euclidean distance
with swap distance (number of non-matching edges) and replacing real-valued
position updates with node-extraction / guided-insertion movements governed
by a three-term favourability score.

Reference:
    Ai, T. J., & Kachitvichyanukul, V. "A particle swarm optimization for
    the vehicle routing problem with simultaneous pickup and delivery", 2009.
    Sayadi, M. K., Ramezanian, R., & Ghaffari-Nasab, N. "A discrete firefly
    meta-heuristic with local search for makespan minimization in permutation
    flow shop scheduling problems", 2010.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)
from logic.src.policies.route_construction.meta_heuristics.firefly_algorithm.params import FAParams


class FASolver:
    """
    Discrete Firefly Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: FAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Pre-instantiate Local Search for reuse
        # Pre-instantiate Local Search for reuse
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Discrete Firefly Algorithm.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        population = [self._build_random_solution() for _ in range(self.params.pop_size)]
        profits = [self._evaluate(f) for f in population]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Pairwise attraction: dimmer firefly moves toward brighter one
            for i in range(self.params.pop_size):
                moved = False
                for j in range(self.params.pop_size):
                    if profits[j] <= profits[i]:
                        continue

                    # Distance r_ij between firefly i and j
                    # In discrete FA, this is often the swap distance
                    r = self._swap_distance(population[i], population[j])

                    # Attractiveness beta = beta0 * exp(-gamma * r^2)
                    # Reference: Yang (2008) Eq (11)
                    beta = self.params.beta0 * np.exp(-self.params.gamma * (r**2))

                    if self.random.random() < beta:
                        # Move firefly i toward j
                        # In discrete case, this means inheriting features from j
                        new_routes = self._attract(population[i], population[j])
                        new_profit = self._evaluate(new_routes)
                        if new_profit > profits[i]:
                            population[i] = new_routes
                            profits[i] = new_profit
                            moved = True

                # Random walk if not attracted or by chance
                if not moved or self.random.random() < self.params.alpha_rnd:
                    rw = self._random_walk(population[i])
                    rw_profit = self._evaluate(rw)
                    if rw_profit > profits[i]:
                        population[i] = rw
                        profits[i] = rw_profit

                # Update global best
                if profits[i] > best_profit:
                    best_routes = copy.deepcopy(population[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.pop_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _swap_distance(self, routes_a: List[List[int]], routes_b: List[List[int]]) -> int:
        """
        Compute discrete swap distance between two routing solutions.

        Defined as the number of edges present in one solution but not the
        other (symmetric edge set difference).

        Args:
            routes_a: First route set.
            routes_b: Second route set.

        Returns:
            Integer swap distance ≥ 0.
        """
        edges_a = self._edge_set(routes_a)
        edges_b = self._edge_set(routes_b)
        return len(edges_a.symmetric_difference(edges_b))

    @staticmethod
    def _edge_set(routes: List[List[int]]) -> set:
        """Extract the set of directed edges (including depot arcs) from routes."""
        edges = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))
        return edges

    def _attract(self, dim_routes: List[List[int]], bright_routes: List[List[int]]) -> List[List[int]]:
        """
        Move dim firefly toward bright firefly via Ai & Kachitvichyanukul (2009)
        node extraction and guided insertion.

        Favourability score per candidate node n:
            score(n) = alpha_p * profit(n) + beta_w * willingness(n) - gamma_c * insertion_cost(n)

        Args:
            dim_routes: Routes of the less bright firefly.
            bright_routes: Routes of the brighter firefly.

        Returns:
            Updated routes for the dim firefly.
        """
        dim_nodes = [n for r in dim_routes for n in r]
        bright_nodes = [n for r in bright_routes for n in r]

        # 1. Extraction: Identify nodes in bright but not in dim (VRPP specific)
        # Or nodes that have different positions.
        # For simplicity and following Ai & Kachitvichyanukul (2009),
        # we extract a set of nodes from the brighter firefly.

        dim_set = set(dim_nodes)
        candidates = [n for n in bright_nodes if n not in dim_set]

        # If no new nodes in bright, pick some nodes from bright to "re-align"
        if not candidates:
            n_extract = self.random.randint(1, max(1, len(bright_nodes) // 4))
            candidates = self.random.sample(bright_nodes, n_extract)

        # 2. Guided Insertion using Favourability Score
        # score(n) = alpha * profit + beta * waste_fraction - gamma * min_insertion_cost
        current_routes = copy.deepcopy(dim_routes)

        # Sort candidates by a simplified favourability to decide insertion order
        # In a full sequential construction, we'd re-calculate after each insertion.
        scored_candidates = []
        for n in candidates:
            profit = self.wastes.get(n, 0.0) * self.R
            willingness = self.wastes.get(n, 0.0) / self.capacity
            cost = self._best_insertion_cost(n, current_routes)

            score = (
                self.params.alpha_profit * profit + self.params.beta_will * willingness - self.params.gamma_cost * cost
            )
            scored_candidates.append((score, n))

        # Select best candidates to insert based on attraction beta (randomized here)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for _, node in scored_candidates:
            # Re-calculate best position in current routes
            best_pos = None
            min_inc_cost = float("inf")
            best_r_idx = -1

            node_waste = self.wastes.get(node, 0.0)

            for r_idx, route in enumerate(current_routes):
                # Capacity check
                if sum(self.wastes.get(nd, 0.0) for nd in route) + node_waste > self.capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = 0 if pos == 0 else route[pos - 1]
                    nxt = 0 if pos == len(route) else route[pos]
                    inc_cost = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]

                    if inc_cost < min_inc_cost:
                        min_inc_cost = inc_cost
                        best_pos = pos
                        best_r_idx = r_idx

            if best_r_idx != -1:
                current_routes[best_r_idx].insert(best_pos, node)  # type: ignore[arg-type]
            else:
                # Add as a new route if profitable or mandatory
                if (
                    node_waste * self.R - (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C > 0
                    or node in self.mandatory_nodes
                ):
                    current_routes.append([node])

        # 3. Apply local search to refine the "attracted" solution
        return self.ls.optimize(current_routes)

    def _partition_flat(self, nodes: List[int]) -> List[List[int]]:
        """Partition flat nodes into feasible routes."""
        routes: List[List[int]] = []
        curr: List[int] = []
        load = 0.0
        for n in nodes:
            w = self.wastes.get(n, 0.0)
            if load + w <= self.capacity:
                curr.append(n)
                load += w
            else:
                if curr:
                    routes.append(curr)
                curr = [n]
                load = w
        if curr:
            routes.append(curr)

        # Mandatory coverage
        visited = {n for r in routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                routes.append([n])
        return routes

    def _best_insertion_cost(self, node: int, routes: List[List[int]]) -> float:
        """
        Compute the minimum insertion cost of a node into existing routes.

        Args:
            node: Node index to insert.
            routes: Existing routes.

        Returns:
            Minimum additional distance required to insert the node.
        """
        min_cost = float("inf")
        for route in routes:
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]
                c = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]
                if c < min_cost:
                    min_cost = c
        # If no routes, cost = depot → node → depot
        if not routes:
            min_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
        return max(0.0, min_cost)

    def _random_walk(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Random walk: remove one node randomly and reinsert greedily.

        Args:
            routes: Current routes.

        Returns:
            Perturbed routes.
        """
        try:
            n_rem = max(3, self.params.n_removal)
            expand_pool = self.params.vrpp
            partial, removed = random_removal(routes, n_rem, self.random)
            if self.params.profit_aware_operators:
                repaired = greedy_profit_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                repaired = greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )

            # Apply comprehensive local search (reusing instance)
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
