"""
DR-ALNS Solver: DRL-controlled Adaptive Large Neighborhood Search.

Combines PPO agent with ALNS for online parameter control.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization import (
    build_nn_routes,
)

from .ppo_agent import DRALNSPPOAgent, DRALNSState


class DRALNSSolver:
    """
    DR-ALNS Solver integrating PPO agent with ALNS.

    The PPO agent learns to:
    1. Select destroy and repair operators
    2. Configure destroy severity
    3. Set acceptance criterion temperature
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        agent: DRALNSPPOAgent,
        max_iterations: int = 100,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DR-ALNS solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1).
            wastes: Dictionary mapping node indices to waste amounts.
            capacity: Vehicle capacity.
            R: Revenue per unit waste.
            C: Cost per unit distance.
            agent: Trained PPO agent.
            max_iterations: Maximum number of iterations.
            mandatory_nodes: List of nodes that must be visited.
            seed: Random seed.
            device: Torch device for agent.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.agent = agent
        self.max_iterations = max_iterations
        self.mandatory_nodes = mandatory_nodes or []
        self.device = device or torch.device("cpu")

        # Move agent to device
        self.agent.to(self.device)
        self.agent.eval()  # Inference mode

        # Problem setup
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(seed)

        # Define destroy operators (matching paper setup)
        self.destroy_ops = [
            self._random_removal,
            self._worst_removal,
            self._cluster_removal,
        ]

        # Define repair operators (matching paper setup)
        self.repair_ops = [
            self._greedy_insertion,
            self._regret_2_insertion,
        ]

        # State tracker
        self.state = DRALNSState()

    def solve(self, initial_routes: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Solve using DR-ALNS.

        Args:
            initial_routes: Optional initial solution.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        # Initialize solution
        routes = self._build_initial_solution() if initial_routes is None else copy.deepcopy(initial_routes)

        current_profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = current_profit
        previous_profit = current_profit

        iterations_since_best = 0
        accepted = True  # Initialize for first iteration
        new_best = False

        for iteration in range(self.max_iterations):
            # Update state
            self.state.update(
                best_profit=best_profit,
                current_profit=current_profit,
                previous_profit=previous_profit,
                new_accepted=accepted,
                new_best=new_best,
                iteration=iteration,
                max_iterations=self.max_iterations,
                iterations_since_best=iterations_since_best,
            )

            # Get state tensor
            state_tensor = self.state.to_tensor(self.device)

            # Select actions using PPO agent
            with torch.no_grad():
                actions, _, _ = self.agent.get_action(state_tensor, deterministic=True)

            # Extract actions
            destroy_idx = actions["destroy"]
            repair_idx = actions["repair"]
            severity_idx = actions["severity"]
            temp_idx = actions["temp"]

            # Convert severity index to percentage (1-10 -> 0.1-1.0)
            destroy_severity = (severity_idx + 1) / 10.0

            # Convert temp index to temperature (1-50 -> 0.1-5.0)
            temperature = (temp_idx + 1) * 0.1

            # Apply destroy and repair operators
            try:
                new_routes = self._apply_operators(routes, destroy_idx, repair_idx, destroy_severity)
                new_profit = self._evaluate(new_routes)
            except Exception:
                # If operator fails, skip this iteration
                continue

            # Acceptance criterion (Simulated Annealing)
            accepted = self._accept_solution(current_profit, new_profit, temperature)

            if accepted:
                previous_profit = current_profit
                routes = new_routes
                current_profit = new_profit

                # Update best solution
                if current_profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = current_profit
                    iterations_since_best = 0
                    new_best = True
                else:
                    iterations_since_best += 1
                    new_best = False
            else:
                iterations_since_best += 1
                new_best = False

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    def _apply_operators(
        self,
        routes: List[List[int]],
        destroy_idx: int,
        repair_idx: int,
        severity: float,
    ) -> List[List[int]]:
        """Apply selected destroy and repair operators."""
        # Calculate number of nodes to remove
        total_nodes = sum(len(route) for route in routes)
        n_remove = 0 if total_nodes == 0 else max(1, int(total_nodes * severity))

        # Apply destroy operator
        destroy_op = self.destroy_ops[destroy_idx]
        partial_routes, removed_nodes = destroy_op(copy.deepcopy(routes), n_remove)

        # Apply repair operator
        repair_op = self.repair_ops[repair_idx]
        repaired_routes = repair_op(partial_routes, removed_nodes)

        return repaired_routes

    def _accept_solution(self, current_profit: float, new_profit: float, temperature: float) -> bool:
        """Simulated Annealing acceptance criterion."""
        if new_profit >= current_profit:
            return True

        if temperature <= 0:
            return False

        delta = new_profit - current_profit  # negative value
        probability = np.exp(delta / temperature)
        return self.random.random() < probability

    # ========================================================================
    # Destroy Operators
    # ========================================================================

    def _random_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Random removal destroy operator."""
        return random_removal(routes, n, rng=self.random)

    def _worst_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Worst removal (most costly) destroy operator."""
        return worst_removal(routes, n, self.dist_matrix)

    def _cluster_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Cluster removal (spatially related) destroy operator."""
        return cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)

    # ========================================================================
    # Repair Operators
    # ========================================================================

    def _greedy_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Greedy insertion repair operator."""
        return greedy_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _regret_2_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Regret-2 insertion repair operator."""
        return regret_2_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _build_initial_solution(self) -> List[List[int]]:
        """Build initial solution using nearest neighbor."""
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate solution profit (revenue - cost)."""
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        cost = self._cost(routes) * self.C
        return revenue - cost

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_cost = 0.0
        for route in routes:
            if not route:
                continue
            # Cost from depot to first node
            total_cost += self.dist_matrix[0, route[0]]
            # Cost between consecutive nodes
            for i in range(len(route) - 1):
                total_cost += self.dist_matrix[route[i], route[i + 1]]
            # Cost from last node to depot
            total_cost += self.dist_matrix[route[-1], 0]
        return total_cost
