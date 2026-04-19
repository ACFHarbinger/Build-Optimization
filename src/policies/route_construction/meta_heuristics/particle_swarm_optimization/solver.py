"""
Particle Swarm Optimization (PSO) for VRPP.

**TRUE PSO IMPLEMENTATION** with inertia-weighted velocity updates.
This rigorously replaces the Sine Cosine Algorithm (SCA), which is
mathematically equivalent to PSO without velocity momentum.

Mathematical Deconstruction of SCA:
    SCA Update: X' = X + r₁·sin(r₂)·|r₃·P - X|
    ≡ PSO Social Term: X' = X + w·(G_best - X) where w = r₁·sin(r₂)·r₃

    Problems with SCA:
    1. sin(r₂) where r₂~U(0,2π) is just a random weight in [-1,1]
    2. No periodicity exploitation (r₂ resampled every iteration)
    3. cos/sin switch is redundant (phase-shifted same distribution)
    4. Expensive transcendental calls with no benefit

    PSO Advantages:
    1. Velocity maintains momentum from previous iterations
    2. Personal best enables individual particle learning
    3. Simpler, faster arithmetic operations
    4. 30+ years of theoretical foundation

Core PSO Algorithm (Kennedy & Eberhart 1995):
    1. Initialize swarm with random positions and velocities
    2. For each iteration:
        a. For each particle i:
            - Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
            - Update position: x(t+1) = x(t) + v(t+1)
            - Evaluate fitness f(x(t+1))
            - Update personal best if f(x(t+1)) > f(pbest_i)
        b. Update global best from all personal bests
        c. Decrease inertia weight w linearly

Encoding:
    - Continuous position vectors in [-1, 1]^n
    - Sigmoid binarization: select nodes where sigmoid(x_i) > 0.5
    - Largest Rank Value (LRV) ordering: sort by x_i descending

Complexity:
    - Time: O(T × N × n) where T = iterations, N = pop_size, n = nodes
    - Space: O(N × n) for swarm + velocities + personal bests

References:
    Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
    Proceedings of ICNN'95 - International Conference on Neural Networks.

    Mirjalili, S. (2016). "SCA: A Sine Cosine Algorithm for solving
    optimization problems." Knowledge-Based Systems.
    [Note: SCA is PSO without velocity - this implementation supersedes it]
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.recreate_repair.greedy import (
    greedy_insertion,
    greedy_profit_insertion,
)

from .params import PSOParams


class PSOSolver:
    """
    Particle Swarm Optimization solver with velocity momentum for VRPP.

    **Replaces SCA** - Proper PSO with all components intact.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: PSOParams,
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
        self.np_rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

        # PSO State: Velocities and Personal Bests
        self.velocities: np.ndarray = np.array([])  # Shape: (pop_size, n_nodes)
        self.personal_bests: np.ndarray = np.array([])  # Shape: (pop_size, n_nodes)
        self.personal_best_fitness: np.ndarray = np.array([])  # Shape: (pop_size,)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

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
        T = self.params.max_iterations

        # Initialize population in continuous space [-1, 1]
        X = self.np_rng.uniform(
            self.params.position_min, self.params.position_max, (self.params.pop_size, self.n_nodes)
        )

        # Initialize velocities (small random values)
        self.velocities = self.np_rng.uniform(
            -self.params.velocity_max, self.params.velocity_max, (self.params.pop_size, self.n_nodes)
        )

        # Decode and evaluate initial population
        routes_pop = [self._decode(x) for x in X]
        profits = np.array([self._evaluate(r) for r in routes_pop])

        # Initialize personal bests (pbest)
        self.personal_bests = X.copy()
        self.personal_best_fitness = profits.copy()

        # Initialize global best (gbest)
        best_idx = int(np.argmax(profits))
        X_best = X[best_idx].copy()
        best_routes = copy.deepcopy(routes_pop[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        # PSO main loop with velocity momentum
        for t in range(T):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Compute dynamic inertia weight (linearly decreasing)
            w = self.params.get_inertia_weight(t)

            # Update each particle using PSO velocity equation
            for i in range(self.params.pop_size):
                # Generate random coefficients for cognitive and social terms
                r1 = self.np_rng.uniform(0, 1, self.n_nodes)
                r2 = self.np_rng.uniform(0, 1, self.n_nodes)

                # PSO Velocity Update: v(t+1) = w*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
                cognitive_velocity = self.params.c1 * r1 * (self.personal_bests[i] - X[i])
                social_velocity = self.params.c2 * r2 * (X_best - X[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity

                # Velocity clamping to prevent divergence
                self.velocities[i] = np.clip(self.velocities[i], -self.params.velocity_max, self.params.velocity_max)

                # Position Update: x(t+1) = x(t) + v(t+1)
                X[i] = X[i] + self.velocities[i]

                # Position clamping to maintain bounds
                X[i] = np.clip(X[i], self.params.position_min, self.params.position_max)

                # Decode and evaluate new position
                routes_pop[i] = self._decode(X[i])
                profits[i] = self._evaluate(routes_pop[i])

                # Update personal best if improved
                if profits[i] > self.personal_best_fitness[i]:
                    self.personal_bests[i] = X[i].copy()
                    self.personal_best_fitness[i] = profits[i]

                # Update global best if improved
                if profits[i] > best_profit:
                    X_best = X[i].copy()
                    best_routes = copy.deepcopy(routes_pop[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=t,
                best_profit=best_profit,
                best_cost=best_cost,
                inertia_weight=w,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _decode(self, x: np.ndarray) -> List[List[int]]:
        """
        Decode a continuous position vector to a discrete routing solution.

        Uses sigmoid binarization + Largest Rank Value (LRV) ordering.
        Same decoding as SCA for fair comparison.

        Steps:
          1. Apply sigmoid binarization: b_j = 1 if sigmoid(x_j) > 0.5
          2. Among selected nodes (b_j=1), order by LRV (sort by x_j descending)
          3. Insert mandatory nodes that were not selected
          4. Build routes via greedy_insertion

        Args:
            x: Continuous position vector of length n_nodes.

        Returns:
            Routing solution as list of routes.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-x))

        mandatory_set = set(self.mandatory_nodes)
        selected_nodes: List[int] = []

        # Always include mandatory nodes
        for node in self.nodes:
            if node in mandatory_set:
                selected_nodes.append(node)

        # Include optional nodes where sigmoid > 0.5
        total_load = sum(self.wastes.get(n, 0.0) for n in selected_nodes)
        optional_sorted = sorted(
            [(sigmoid[idx], self.nodes[idx]) for idx in range(self.n_nodes) if self.nodes[idx] not in mandatory_set],
            reverse=True,
        )

        for _, node in optional_sorted:
            w = self.wastes.get(node, 0.0)
            if sigmoid[self.nodes.index(node)] > 0.5 and total_load + w <= self.capacity:
                selected_nodes.append(node)
                total_load += w

        if not selected_nodes:
            return []

        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp
        if use_profit:
            return greedy_profit_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return greedy_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate routing solution fitness (net profit).

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: Routing solution.

        Returns:
            Net profit (higher is better).
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
