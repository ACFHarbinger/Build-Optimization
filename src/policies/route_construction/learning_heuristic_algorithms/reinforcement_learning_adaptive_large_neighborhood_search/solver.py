"""
RL-ALNS: Reinforcement Learning-augmented Adaptive Large Neighborhood Search.

This module implements an ALNS solver that uses various online RL algorithms
to intelligently select destroy/repair operators during the search process.

Based on the research: "Online Reinforcement Learning for Inference-Time
Operator Selection in the Stochastic Multi-Period Capacitated Vehicle Routing Problem"
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.destroy_ruin.shaw import shaw_removal
from logic.src.policies.helpers.operators.destroy_ruin.string import string_removal
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    DiscountedUCBBandit as DiscountedUCBAgent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    EXP3Agent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    SlidingWindowUCBBandit as SlidingWindowUCBAgent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    ThompsonSamplingBandit as ThompsonSamplingAgent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    UCBBandit as UCB1Agent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.td_learning import (
    ExpectedSarsaAgent as ExpectedSARSAAgent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.td_learning import (
    QLearningAgent,
)
from logic.src.policies.helpers.reinforcement_learning.agents.td_learning import (
    SarsaAgent as SARSAAgent,
)
from logic.src.policies.helpers.reinforcement_learning.features.state import StateFeatureExtractor
from logic.src.policies.helpers.reinforcement_learning.reward.shaping import AdaptiveRewardShaper, RewardShaper

from .params import RLALNSParams


class RLALNSSolver:
    """
    RL-ALNS solver with multiple RL algorithms for operator selection.

    Supports:
    - Multi-Armed Bandits: UCB1, D-UCB, SW-UCB, Thompson Sampling, EXP3
    - Temporal Difference Learning: Q-Learning, SARSA, Expected SARSA
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RLALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize RL-ALNS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes/demands.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: RL-ALNS parameters.
            mandatory_nodes: List of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes if mandatory_nodes is not None else []
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.np_random = np.random.default_rng(params.seed)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Initialize operators
        self._init_operators()

        # Initialize RL components
        self.feature_extractor = StateFeatureExtractor(
            progress_thresholds=params.progress_thresholds,  # type: ignore[arg-type]
            stagnation_thresholds=params.stagnation_thresholds,  # type: ignore[arg-type]
            diversity_thresholds=params.diversity_thresholds,  # type: ignore[arg-type]
        )

        if params.adaptive_rewards:
            self.reward_shaper = AdaptiveRewardShaper(
                best_improvement_reward=params.reward_new_global_best,
                local_improvement_reward=params.reward_improved_current,
                accepted_reward=params.reward_accepted_worse,
                rejected_reward=params.reward_rejected,
            )
        else:
            self.reward_shaper = RewardShaper(  # type: ignore[assignment]
                best_improvement_reward=params.reward_new_global_best,
                local_improvement_reward=params.reward_improved_current,
                accepted_reward=params.reward_accepted_worse,
                rejected_reward=params.reward_rejected,
            )

        # Initialize RL agent
        self.rl_agent = self._create_rl_agent()

        # Tracking
        self.improvement_history: List[float] = []
        self.operator_performance: List[Dict] = []

    def _init_operators(self) -> None:
        """
        Initialize the portfolio of destroy and repair operators.

        Sets up the destroy_ops and repair_ops lists with lambda wrappers
        that call the underlying operator functions with appropriate arguments.
        Provides a mapping for action indices to operator combinations.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_profit_removal(r, n, self.dist_matrix, self.wastes, self.R, self.C),
                lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
                lambda r, n: shaw_profit_removal(r, n, self.dist_matrix, self.wastes, self.R, self.C),
                lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
            ]
        else:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_removal(r, n, self.dist_matrix),
                lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
                lambda r, n: shaw_removal(r, n, self.dist_matrix),
                lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
            ]

        self.destroy_names = ["Random", "Worst", "Cluster", "Shaw", "String"]

        if use_profit:
            self.repair_ops = [
                lambda r, n: greedy_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
                lambda r, n: regret_2_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
            ]
        else:
            self.repair_ops = [
                lambda r, n: greedy_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
                lambda r, n: regret_2_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                ),
            ]
        self.repair_names = ["Greedy", "Regret-2"]

        # Calculate total number of possible (destroy, repair) combinations
        self.n_actions = len(self.destroy_ops) * len(self.repair_ops)

    def _create_rl_agent(self):
        """
        Factory method to create the appropriate RL agent based on configuration.

        Supports a wide variety of MAB and TD-learning agents through a unified interface.

        Returns:
            RLAgent: An instance of a bandit or TD-learning agent.

        Raises:
            ValueError: If the configured algorithm is not recognized.
        """
        algo = self.params.rl_algorithm.lower()
        seed = int(self.np_random.integers(1000000))

        if algo == "ucb1":
            return UCB1Agent(n_arms=self.n_actions, c=self.params.ucb_c, seed=seed)

        elif algo == "discounted_ucb":
            return DiscountedUCBAgent(
                n_arms=self.n_actions,
                c=self.params.ucb_c,
                gamma=self.params.ucb_gamma,
                seed=seed,
            )

        elif algo == "sliding_window_ucb":
            return SlidingWindowUCBAgent(
                n_arms=self.n_actions,
                window_size=self.params.ucb_window_size,
                c=self.params.ucb_c,
                seed=seed,
            )

        elif algo == "thompson_sampling":
            return ThompsonSamplingAgent(
                n_arms=self.n_actions,
                alpha_prior=self.params.ts_alpha_prior,
                beta_prior=self.params.ts_beta_prior,
                seed=seed,
            )

        elif algo == "exp3":
            return EXP3Agent(n_arms=self.n_actions, gamma=self.params.exp3_gamma, seed=seed)

        elif algo == "q_learning":
            n_states = 27  # 3x3x3 discretized states
            return QLearningAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        elif algo == "sarsa":
            n_states = 27
            return SARSAAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        elif algo == "expected_sarsa":
            n_states = 27
            return ExpectedSARSAAgent(
                n_states=n_states,
                n_actions=self.n_actions,
                alpha=self.params.alpha,
                gamma=self.params.gamma,
                epsilon=self.params.epsilon,
                epsilon_decay=self.params.epsilon_decay,
                epsilon_min=self.params.epsilon_min,
                seed=seed,
            )

        else:
            raise ValueError(f"Unknown RL algorithm: {algo}")

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run RL-ALNS algorithm.

        Args:
            initial_solution: Initial solution routes (optional).

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        # Initialize solution
        current_routes = [r[:] for r in initial_solution] if initial_solution else self._build_initial_solution()
        best_routes = [r[:] for r in current_routes]

        # Calculate initial costs
        best_cost = self._calculate_cost(best_routes)
        current_cost = best_cost
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in best_routes for node in route)
        best_profit = collected_revenue - (best_cost * self.C)
        current_profit = best_profit

        # Initialize search state
        T = self.params.start_temp
        start_time = time.process_time()
        stagnation_count = 0
        self.improvement_history = []

        # Main loop
        for iteration in range(self.params.max_iterations):
            # Check time limit
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Extract state features
            features = self.feature_extractor.extract_features(
                routes=current_routes,
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                iteration=iteration,
                max_iterations=self.params.max_iterations,
                current_cost=current_cost,
                best_cost=best_cost,
                stagnation_count=stagnation_count,
                improvement_history=self.improvement_history,
            )

            # Discretize state for tabular RL
            state_tuple = self.feature_extractor.discretize_state(
                iteration=iteration,
                max_iterations=self.params.max_iterations,
                stagnation_count=stagnation_count,
                diversity=features["route_diversity"],
            )
            state_index = self.feature_extractor.state_to_index(state_tuple)

            # Select operator pair using RL agent
            action = self.rl_agent.select_action(state_index, self.np_random)
            d_idx, r_idx = self._action_to_operators(action)

            # Apply destroy-repair
            new_routes = self._apply_operators(current_routes, d_idx, r_idx)

            # Evaluate new solution
            new_cost = self._calculate_cost(new_routes)
            collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in new_routes for node in route)
            new_profit = collected_revenue - (new_cost * self.C)

            # Acceptance (Simulated Annealing)
            delta = new_profit - current_profit
            accepted = False

            if delta > -1e-6:
                accepted = True
            else:
                prob = math.exp(delta / T) if T > 0 else 0
                if self.random.random() < prob:
                    accepted = True

            # Calculate reward
            reward = self.reward_shaper.calculate_reward(
                new_cost=new_cost,
                current_cost=current_cost,
                best_cost=best_cost,
                accepted=accepted,
            )

            # Update RL agent
            next_state_index = state_index  # Simple approach for now
            self.rl_agent.update(state_index, action, reward, next_state_index, False)

            # Accept/reject solution
            if accepted:
                current_routes = new_routes
                current_cost = new_cost
                current_profit = new_profit

                if current_profit > best_profit + 1e-6:
                    best_routes = [r[:] for r in current_routes]
                    best_profit = current_profit
                    best_cost = current_cost
                    stagnation_count = 0
                    self.improvement_history.append(1.0)
                else:
                    stagnation_count += 1
                    self.improvement_history.append(0.0)
            else:
                stagnation_count += 1
                self.improvement_history.append(0.0)

            # Cool temperature
            T *= self.params.cooling_rate

            # Record for visualization
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_cost=best_profit,
                current_cost=current_profit,
                temperature=T,
                accepted=int(accepted),
                destroy_op=self.destroy_names[d_idx],
                repair_op=self.repair_names[r_idx],
                reward=reward,
            )

        return best_routes, best_profit, best_cost

    def _apply_operators(self, routes: List[List[int]], d_idx: int, r_idx: int) -> List[List[int]]:
        """
        Execute a destroy-repair iteration using selected operators.

        Args:
            routes (List[List[int]]): Current solution routes.
            d_idx (int): Index of the destroy operator to use.
            r_idx (int): Index of the repair operator to use.

        Returns:
            List[List[int]]: The newly generated candidate solution.
        """
        # Determine removal size dynamically based on solution scale
        current_n_nodes = sum(len(route) for route in routes)

        if current_n_nodes == 0:
            n_remove = 0
        else:
            # Ensure we remove at least 2 nodes but no more than configured percentage
            lower_bound = min(current_n_nodes, 2)
            max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
            upper_bound = max(lower_bound + 1, max_pct_remove)
            upper_bound = min(upper_bound, current_n_nodes)
            n_remove = self.random.randint(lower_bound, upper_bound)

        # 1. Destroy Step: remove a subset of nodes from the current solution
        destroy_op = self.destroy_ops[d_idx]
        partial_routes, removed = destroy_op(copy.deepcopy(routes), n_remove)

        # 2. Repair Step: reinsert removed nodes back into the partial routes
        repair_op = self.repair_ops[r_idx]
        new_routes = repair_op(partial_routes, removed)

        return new_routes

    def _action_to_operators(self, action: int) -> Tuple[int, int]:
        """
        Map a single flat action index back to its component (destroy, repair) indices.

        Args:
            action (int): Flat action index in range [0, n_actions).

        Returns:
            Tuple[int, int]: (destroy_operator_index, repair_operator_index).
        """
        n_repair = len(self.repair_ops)
        d_idx = action // n_repair
        r_idx = action % n_repair
        return d_idx, r_idx

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate the total routing cost for a set of routes.

        Args:
            routes (List[List[int]]): The routes to evaluate.

        Returns:
            float: Total distance covered by all routes.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depot -> First node
            total += float(self.dist_matrix[0, route[0]])
            # Between nodes
            for i in range(len(route) - 1):
                total += float(self.dist_matrix[route[i], route[i + 1]])
            # Last node -> Depot
            total += float(self.dist_matrix[route[-1], 0])
        return total

    def _build_initial_solution(self) -> List[List[int]]:
        """
        Construct an initial feasible solution using the Nearest Neighbor heuristic.

        Returns:
            List[List[int]]: Initial set of routes.
        """
        routes = build_nn_routes(
            nodes=list(self.wastes.keys()),
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return routes

    def get_statistics(self) -> Dict:
        """Get solver statistics."""
        stats = {
            "algorithm": self.params.rl_algorithm,
            "rl_agent_stats": self.rl_agent.get_statistics(),
            "reward_config": self.reward_shaper.get_reward_config(),
            "total_iterations": len(self.improvement_history),
            "total_improvements": sum(self.improvement_history),
        }
        return stats
