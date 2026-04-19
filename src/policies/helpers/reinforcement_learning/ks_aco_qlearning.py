"""
Enhanced K-Sparse ACO Solver with Q-Learning for Local Search Operator Selection.

This module extends the standard K-Sparse ACO with Q-Learning reinforcement learning
to dynamically select the most effective local search operators for route improvement.

Reference:
    Watkins & Dayan, "Q-Learning", Machine Learning, 1992.
    Gambardella & Dorigo, "Ant Colony System", IEEE Trans., 1997.
"""

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager
from logic.src.policies.helpers.operators.solution_initialization import build_nn_routes
from logic.src.policies.helpers.reinforcement_learning.agents.td_learning import QLearningAgent
from logic.src.policies.helpers.reinforcement_learning.features.state import StateFeatureExtractor
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction import (
    SolutionConstructor,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones import (
    SparsePheromoneTau,
)


class KSparseACOQLSolver:
    """
    Enhanced K-Sparse ACO with Q-Learning for local search operator selection.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: KSACOParams,
        rl_params: Any,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initialize enhanced ACO-QL solver."""
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.epsilon_decay_step = rl_params.qlearning_epsilon_decay_step
        self.improvement_threshold = rl_params.qlearning_improvement_thresholds[0]

        self.n_nodes = len(dist_matrix)
        self.nodes = list(range(1, self.n_nodes))

        # Pheromone and heuristic
        self.eta = np.zeros_like(dist_matrix, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)

        nn_cost = self._nearest_neighbor_cost()
        self.tau_0 = (
            params.tau_0
            if params.tau_0 is not None and params.tau_0 > 0
            else 1.0 / (self.n_nodes * nn_cost)
            if nn_cost > 0
            else params.tau_max
        )

        self.pheromone = SparsePheromoneTau(self.n_nodes, params.k_sparse, self.tau_0, params.tau_min, params.tau_max)

        # Local search
        self.ls_manager = LocalSearchManager(
            dist_matrix, wastes, capacity, R, C, rl_params.qlearning_improvement_thresholds[1], seed=params.seed
        )

        # Candidate lists
        self.candidate_lists = self._build_candidate_lists()

        # Constructor
        self.constructor = SolutionConstructor(
            dist_matrix,
            wastes,
            capacity,
            self.pheromone,
            self.eta,
            self.candidate_lists,
            self.nodes,
            params,
            self.tau_0,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
        )

        # Q-Learning selector for local search operators
        # Full operator set across all categories
        self.ls_operators = [
            # Exchange operators
            "or_opt",
            "cross_exchange",
            "lambda_interchange",
            "ejection_chain",
            # Route operators
            "2opt_star",
            "swap_star",
            "2opt_intra",
            "3opt_intra",
            # Move operators
            "relocate",
            "swap",
            # Heuristic operators
            "lkh_refinement",
        ]

        # Initialize RL components
        self.feature_extractor = StateFeatureExtractor()  # Default thresholds fine for now
        self.n_actions = len(self.ls_operators)
        self.agent = QLearningAgent(
            n_states=27,  # 3*3*3 discretized states
            n_actions=self.n_actions,
            alpha=rl_params.qlearning_alpha,
            gamma=rl_params.qlearning_gamma,
            epsilon=rl_params.qlearning_epsilon,
            epsilon_decay=rl_params.qlearning_epsilon_decay,
            epsilon_min=rl_params.qlearning_epsilon_min,
        )
        self.agent_rng = np.random.default_rng(params.seed)

        # Tracking
        self.improvement_history: Deque[float] = deque(maxlen=rl_params.qlearning_history_size)

    def _nearest_neighbor_cost(self) -> float:
        """Compute nearest neighbor heuristic cost."""
        visited = set([0])
        current = 0
        cost = 0.0
        for _ in range(len(self.nodes)):
            best_next = None
            best_dist = float("inf")
            for node in self.nodes:
                if node not in visited:
                    d = self.dist_matrix[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_next = node
            if best_next is not None:
                cost += best_dist
                visited.add(best_next)
                current = best_next
        cost += self.dist_matrix[current][0]
        return cost

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbor candidate lists."""
        candidates: Dict[int, List[int]] = {}
        k = min(self.params.k_sparse, len(self.nodes))

        for i in range(self.n_nodes):
            distances = [(self.dist_matrix[i][j], j) for j in range(self.n_nodes) if j != i]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]

        return candidates

    def _initialize_with_nn_heuristic(self) -> Tuple[List[List[int]], float, float]:
        """
        Initialize solution using Nearest Neighbor heuristic.

        Creates geographically compact routes using build_nn_routes and applies
        local search to improve the initial solution.

        Returns:
            Tuple of (routes, profit, cost) for the NN-initialized solution.
        """
        # Build initial routes using NN heuristic
        nn_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes if self.mandatory_nodes else [],
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.constructor.random,
        )

        # Apply local search to improve NN solution
        if self.params.local_search and nn_routes:
            nn_routes = self._q_learning_local_search(nn_routes, iteration=0)

        # Calculate metrics
        cost = self._calculate_cost(nn_routes)
        revenue = sum(self.wastes.get(n, 0) * self.R for r in nn_routes for n in r)
        profit = revenue - cost

        return nn_routes, profit, cost

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run enhanced ACO with Q-Learning local search.

        The algorithm is seeded with a Nearest Neighbor heuristic solution
        to provide a strong starting point for the ACO search.
        """
        # Initialize with NN heuristic for strong starting solution
        best_routes, best_profit, best_cost = self._initialize_with_nn_heuristic()
        start_time = time.process_time()

        # Update pheromones based on NN solution
        if best_routes:
            self._global_pheromone_update(best_routes, best_cost)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            iteration_best_routes: List[List[int]] = []
            iteration_best_profit = float("-inf")
            iteration_best_cost = float("inf")

            # Ant construction
            for _ in range(self.params.n_ants):
                routes = self.constructor.construct()

                # Q-Learning local search
                if self.params.local_search:
                    routes = self._q_learning_local_search(routes, iteration)

                cost = self._calculate_cost(routes)
                revenue = sum(self.wastes.get(n, 0) * self.R for r in routes for n in r)
                profit = revenue - cost

                if profit > iteration_best_profit:
                    iteration_best_profit = profit
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # Update global best
            if iteration_best_profit > best_profit:
                improvement = (iteration_best_profit - best_profit) / (abs(best_profit) + 1e-6)
                self.improvement_history.append(improvement)
                best_profit = iteration_best_profit
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes
            else:
                self.improvement_history.append(0.0)

            # Global pheromone update
            self._global_pheromone_update(best_routes, best_cost)

            # Visualization
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iteration_best_profit,
                epsilon=self.agent.epsilon,
            )

        return best_routes, best_profit, best_cost

    def _q_learning_local_search(self, routes: List[List[int]], iteration: int) -> List[List[int]]:
        """
        Execute Q-Learning guided local search on a set of routes.

        This method discretizes the current search state and uses a Q-Learning
        agent to select the most promising local search operator.

        Args:
            routes (List[List[int]]): Initial routes for local search.
            iteration (int): Current global iteration of the ACO algorithm.

        Returns:
            List[List[int]]: Improved routes after local search.
        """
        initial_cost = self._calculate_cost(routes)
        # Calculate a normalized route quality metric for state representation
        route_quality = 1.0 / (1.0 + initial_cost / 1000.0)

        # Step 1: Discretize current state (iteration phase, stagnation, quality)
        state_tuple = self.feature_extractor.discretize_state(
            iteration,
            self.params.max_iterations,
            0,  # Stagnation count not tracked at this level
            route_quality,
        )
        state = self.feature_extractor.state_to_index(state_tuple)

        # Step 2: Iteratively apply selected local search moves
        for _ in range(self.params.local_search_iterations):
            # Select operator index via the Q-Learning agent's epsilon-greedy policy
            op_idx = self.agent.select_action(state, self.agent_rng)
            operator_name = self.ls_operators[op_idx]

            # Apply the selected operator to the current solution
            improved = self._apply_operator(operator_name)

            # Step 3: Calculate reward based on cost improvement
            new_routes = self.ls_manager.get_routes()
            new_cost = self._calculate_cost(new_routes)
            cost_improvement = initial_cost - new_cost

            # Use a sparse reward structure: significant bonus for improvement,
            # slight penalty for stagnation to encourage activity.
            reward = 10.0 if cost_improvement > self.improvement_threshold else (-1.0 if not improved else 0.0)

            # Step 4: Transition to next state
            new_quality = 1.0 / (1.0 + new_cost / 1000.0)
            next_state_tuple = self.feature_extractor.discretize_state(
                iteration, self.params.max_iterations, 0, new_quality
            )
            next_state = self.feature_extractor.state_to_index(next_state_tuple)

            # Step 5: Update Q-values using the Standard Q-Learning rule:
            # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
            self.agent.update(state, op_idx, reward, next_state, False)

            # Prepare for next local search step
            state = next_state
            if improved:
                initial_cost = new_cost

        # Decay exploration rate regularly to shift from exploration to exploitation
        if iteration % self.epsilon_decay_step == 0:
            self.agent.decay_epsilon()

        return self.ls_manager.get_routes()

    def _apply_operator(self, operator_name: str) -> bool:
        """
        Apply the specified local search operator.

        Args:
            operator_name: Name of the operator to apply.

        Returns:
            True if the operator improved the solution.
        """
        # Exchange operators
        if operator_name == "or_opt":
            return self.ls_manager.or_opt(chain_len=2)
        elif operator_name == "cross_exchange":
            return self.ls_manager.cross_exchange_op(max_seg_len=2)
        elif operator_name == "lambda_interchange":
            return self.ls_manager.lambda_interchange_op(lambda_max=2)
        elif operator_name == "ejection_chain":
            return self.ls_manager.ejection_chain_op(max_depth=3)

        # Route operators
        elif operator_name == "2opt_star":
            return self.ls_manager.two_opt_star()
        elif operator_name == "swap_star":
            return self.ls_manager.swap_star()
        elif operator_name == "2opt_intra":
            return self.ls_manager.two_opt_intra()
        elif operator_name == "3opt_intra":
            return self.ls_manager.three_opt_intra()

        # Move operators
        elif operator_name == "relocate":
            return self.ls_manager.relocate()
        elif operator_name == "swap":
            return self.ls_manager.swap()

        # Heuristic operators
        elif operator_name == "lkh_refinement":
            return self.ls_manager.lkh_refinement()

        else:
            # Unknown operator
            return False

    def _global_pheromone_update(self, best_routes: List[List[int]], best_cost: float):
        """ACS global pheromone update."""
        if not best_routes or best_cost <= 0:
            return

        self.pheromone.evaporate_all(self.params.rho)
        delta = self.params.elitist_weight / best_cost

        for route in best_routes:
            if not route:
                continue
            self.pheromone.deposit_edge(0, route[0], delta)
            for k in range(len(route) - 1):
                self.pheromone.deposit_edge(route[k], route[k + 1], delta)
            self.pheromone.deposit_edge(route[-1], 0, delta)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total * self.C
