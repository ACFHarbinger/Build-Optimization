"""
Particle Swarm Optimization with Velocity Momentum for VRPP.

**TRUE PSO IMPLEMENTATION** with inertia-weighted velocity updates.
This replaces the deceptive "Firefly Algorithm" which lacked velocity momentum.

Core PSO Algorithm (Kennedy & Eberhart 1995, Ai & Kachitvichyanukul 2009):
    1. Initialize swarm of particles (solutions) with random velocities
    2. For each iteration:
        a. For each particle i:
            - Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))
            - Update position: x(t+1) = x(t) + v(t+1)
            - Evaluate fitness f(x(t+1))
            - Update personal best if f(x(t+1)) > f(pbest_i)
        b. Update global best from all personal bests
        c. Decrease inertia weight w linearly

Velocity in Discrete Space (Ai & Kachitvichyanukul 2009):
    - Velocity magnitude → mutation strength
    - Cognitive term (pbest - x) → reintroduce nodes from personal best
    - Social term (gbest - x) → reintroduce nodes from global best
    - Inertia term w*v → preserve previous move direction

Complexity:
    - Time: O(T × N × n²) where T = iterations, N = pop_size, n = nodes
    - Space: O(N × n) for swarm + velocities + personal bests
    - Velocity update: O(n) per particle (node set differences)

References:
    Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
    Proceedings of ICNN'95 - International Conference on Neural Networks.

    Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
    for the vehicle routing problem with simultaneous pickup and delivery."
    Computers & Operations Research, 36(6), 1693-1702.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import greedy_insertion, greedy_profit_insertion, random_removal
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import DistancePSOParams


class DistancePSOSolver:
    """
    Particle Swarm Optimization solver with velocity momentum for VRPP.

    **TRUE PSO** with inertia-weighted velocity updates (Kennedy & Eberhart 1995).
    Each particle maintains:
    - Current position x(t): routing solution
    - Velocity v(t): tendency to change solution (represented as node sets)
    - Personal best pbest: best solution this particle has found
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: DistancePSOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the PSO solver with velocity momentum.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: PSO configuration parameters.
            mandatory_nodes: Nodes that must be visited.
        """
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

        # PSO State: Velocities and Personal Bests
        self.velocities: List[Set[int]] = []  # Velocity as set of nodes to add/remove
        self.personal_bests: List[List[List[int]]] = []  # pbest for each particle
        self.personal_best_fitness: List[float] = []  # f(pbest) for each particle

        # Initialize Local Search once for reuse
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            time_limit=self.params.time_limit,
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

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Particle Swarm Optimization with velocity momentum.

        Implements standard PSO algorithm (Kennedy & Eberhart 1995):
        1. Initialize swarm with random positions and velocities
        2. Iteratively update velocities and positions
        3. Track personal and global bests

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize swarm (particle positions)
        swarm = [self._initialize_particle() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate(particle) for particle in swarm]

        # Initialize velocities (empty sets = no movement initially)
        self.velocities = [set() for _ in range(self.params.population_size)]

        # Initialize personal bests (pbest)
        self.personal_bests = [copy.deepcopy(particle) for particle in swarm]
        self.personal_best_fitness = list(fitness_values)

        # Track global best (gbest)
        best_idx = int(np.argmax(fitness_values))
        best_routes = copy.deepcopy(swarm[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # PSO main loop with velocity momentum
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Compute dynamic inertia weight (linearly decreasing)
            inertia_weight = self.params.get_inertia_weight(iteration)

            # Update each particle using PSO velocity equation
            for i in range(self.params.population_size):
                # Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
                new_velocity = self._update_velocity(
                    current_position=swarm[i],
                    current_velocity=self.velocities[i],
                    personal_best=self.personal_bests[i],
                    global_best=best_routes,
                    inertia_weight=inertia_weight,
                )
                self.velocities[i] = new_velocity

                # Update position: x(t+1) = x(t) + v(t+1)
                new_position = self._apply_velocity(swarm[i], new_velocity)
                new_fitness = self._evaluate(new_position)

                # Accept new position
                swarm[i] = new_position
                fitness_values[i] = new_fitness

                # Update personal best if improved
                if new_fitness > self.personal_best_fitness[i]:
                    self.personal_bests[i] = copy.deepcopy(new_position)
                    self.personal_best_fitness[i] = new_fitness

                # Update global best if improved
                if new_fitness > best_profit:
                    best_routes = copy.deepcopy(new_position)
                    best_profit = new_fitness
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Particle Initialization
    # ------------------------------------------------------------------

    def _initialize_particle(self) -> List[List[int]]:
        """
        Initialize a single particle using nearest-neighbor heuristic.

        Randomized node ordering creates diverse initial swarm distribution.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for nearest-neighbor construction.
        """
        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return routes

    # ------------------------------------------------------------------
    # PSO Velocity Update Methods
    # ------------------------------------------------------------------

    def _update_velocity(
        self,
        current_position: List[List[int]],
        current_velocity: Set[int],
        personal_best: List[List[int]],
        global_best: List[List[int]],
        inertia_weight: float,
    ) -> Set[int]:
        """
        Update particle velocity using PSO equation (discrete adaptation).

        PSO Velocity Equation (Kennedy & Eberhart 1995):
            v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))

        Discrete Interpretation (Ai & Kachitvichyanukul 2009):
            - Velocity = set of nodes representing movement tendency
            - Inertia term w*v(t): preserve fraction of current velocity
            - Cognitive term c₁*(pbest - x): nodes in pbest but not in x
            - Social term c₂*(gbest - x): nodes in gbest but not in x

        Args:
            current_position: Current particle position x(t).
            current_velocity: Current velocity v(t) as node set.
            personal_best: Personal best position pbest.
            global_best: Global best position gbest.
            inertia_weight: Inertia weight w ∈ [0,1].

        Returns:
            New velocity v(t+1) as set of nodes.

        Complexity: O(n) for set operations.
        """
        # Extract node sets
        current_nodes = self._get_node_set(current_position)
        pbest_nodes = self._get_node_set(personal_best)
        gbest_nodes = self._get_node_set(global_best)

        # Inertia term: w * v(t) - probabilistically keep nodes in current velocity
        inertia_contribution = {node for node in current_velocity if self.random.random() < inertia_weight}

        # Cognitive term: c₁ * r₁ * (pbest - x) - nodes in pbest but not in current
        cognitive_difference = pbest_nodes - current_nodes
        r1 = self.random.random()
        cognitive_prob = self.params.c1 * r1
        cognitive_contribution = {node for node in cognitive_difference if self.random.random() < cognitive_prob}

        # Social term: c₂ * r₂ * (gbest - x) - nodes in gbest but not in current
        social_difference = gbest_nodes - current_nodes
        r2 = self.random.random()
        social_prob = self.params.c2 * r2
        social_contribution = {node for node in social_difference if self.random.random() < social_prob}

        # Combine all velocity components
        new_velocity = inertia_contribution | cognitive_contribution | social_contribution

        return new_velocity

    def _apply_velocity(self, current_position: List[List[int]], velocity: Set[int]) -> List[List[int]]:
        """
        Apply velocity to current position to get new position.

        Position Update: x(t+1) = x(t) + v(t+1)

        In discrete space:
        1. Velocity nodes = nodes to add to solution
        2. Remove random nodes (destroy)
        3. Insert velocity nodes greedily (repair)
        4. Apply local search refinement

        Args:
            current_position: Current routing solution x(t).
            velocity: Velocity vector v(t+1) as node set.

        Returns:
            New routing solution x(t+1).

        Complexity: O(n²) for insertion + local search.
        """
        if not velocity:
            # No velocity → small random perturbation for exploration
            return self._random_walk(current_position)

        # Extract current nodes
        current_nodes = self._get_node_set(current_position)

        # Separate velocity into nodes to add vs nodes already present
        nodes_to_add = [n for n in velocity if n not in current_nodes]

        if not nodes_to_add:
            # All velocity nodes already in solution → small perturbation
            return self._random_walk(current_position)

        # Destroy: remove nodes to make room (proportional to velocity magnitude)
        n_remove = min(len(nodes_to_add), max(3, int(len(velocity) * self.params.velocity_to_mutation_rate)))

        try:
            partial_routes, removed_nodes = random_removal(current_position, n_remove, self.random)
        except Exception:
            partial_routes = copy.deepcopy(current_position)
            removed_nodes = []

        # Repair: insert velocity nodes + removed nodes greedily
        all_nodes_to_insert = nodes_to_add + [n for n in removed_nodes if n not in velocity]

        # Score and sort nodes by attraction (same as original attract_toward logic)
        scored_nodes = []
        for node in all_nodes_to_insert:
            profit = self.wastes.get(node, 0.0) * self.R
            fill_level = self.wastes.get(node, 0.0)
            insertion_cost = self._compute_best_insertion_cost(node, partial_routes)
            score = (
                self.params.alpha_profit * profit
                + self.params.beta_will * fill_level
                - self.params.gamma_cost * insertion_cost
            )
            scored_nodes.append((score, node))

        scored_nodes.sort(reverse=True)
        sorted_nodes = [node for _, node in scored_nodes]

        # Greedy insertion
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp
        try:
            if use_profit:
                new_routes = greedy_profit_insertion(
                    partial_routes,
                    sorted_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                new_routes = greedy_insertion(
                    partial_routes,
                    sorted_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            # Apply local search refinement
            return self.ls.optimize(new_routes)
        except Exception:
            return copy.deepcopy(current_position)

    @staticmethod
    def _get_node_set(routes: List[List[int]]) -> Set[int]:
        """
        Extract set of nodes visited in routing solution.

        Args:
            routes: List of routes.

        Returns:
            Set of node indices.

        Complexity: O(n).
        """
        return {node for route in routes for node in route}

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _compute_best_insertion_cost(self, node: int, routes: List[List[int]]) -> float:
        """
        Compute minimum insertion cost for a node into existing routes.

        Finds the position that minimizes additional distance when inserting node.

        Args:
            node: Node index to insert.
            routes: Existing routes.

        Returns:
            Minimum additional distance required.

        Complexity: O(n) for trying all insertion positions.
        """
        min_cost = float("inf")
        for route in routes:
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]
                cost = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]
                if cost < min_cost:
                    min_cost = cost

        # If no routes exist, cost = depot → node → depot
        if not routes:
            min_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]

        return max(0.0, min_cost)

    def _random_walk(self, particle: List[List[int]]) -> List[List[int]]:
        """
        Apply random walk exploration to particle.

        Removes random nodes and reinserts them greedily. This maintains
        swarm diversity and prevents premature convergence.

        Args:
            particle: Current particle (routing solution).

        Returns:
            Perturbed particle after random walk.

        Complexity: O(n²) for destroy-repair operation.
        """
        try:
            n_remove = max(3, self.params.n_removal)
            partial_routes, removed_nodes = random_removal(particle, n_remove, self.random)
            if self.params.profit_aware_operators:
                repaired_routes = greedy_profit_insertion(
                    partial_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                repaired_routes = greedy_insertion(
                    partial_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            # Apply local search refinement
            return self.ls.optimize(repaired_routes)
        except Exception:
            return copy.deepcopy(particle)

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate particle fitness (net profit).

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: Routing solution.

        Returns:
            Net profit (higher is better).

        Complexity: O(n) for route traversal.
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance traveled.

        Complexity: O(n) for route traversal.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
