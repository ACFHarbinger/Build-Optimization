"""
Enhanced Adaptive Large Neighborhood Search (ALNS) with SARSA Reinforcement Learning.

This module implements an advanced ALNS solver that uses SARSA (State-Action-Reward-State-Action)
reinforcement learning to intelligently select destroy/repair operators. It includes all available
operators from the codebase:

Destroy Operators:
    - Random removal
    - Worst removal
    - Cluster removal
    - Shaw removal (relatedness-based)
    - String removal (contiguous segments)
    - Unstringing Type I-IV (GENIUS removal)

Repair Operators:
    - Greedy insertion
    - Regret-2 insertion
    - Regret-k insertion (extended)
    - Greedy blink insertion (randomized)
    - Stringing Type I-IV (GENIUS reinsertion)

Perturbation Operators:
    - Route shuffling
    - Node sequence reversal
    - Random restart from strong perturbation

Unstringing and Stringing Operators (US):
    - Type I: Single string removal/insertion
    - Type II: Multiple string removal/insertion
    - Type III: Propagating string removal/insertion
    - Type IV: Clustered string removal/insertion

Reference:
    Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd Ed., 2018.
    Pisinger & Ropke, "A general heuristic for vehicle routing problems", 2007.
"""

import copy
import random
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import build_greedy_routes
from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal as cluster_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    random_removal as random_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    shaw_profit_removal as shaw_profit_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    shaw_removal as shaw_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    string_removal as string_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    worst_profit_removal as worst_profit_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    worst_removal as worst_removal_op,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion import (
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
)
from logic.src.policies.helpers.operators.perturbation_shaking import kick as kick_op
from logic.src.policies.helpers.operators.perturbation_shaking import kick_profit as kick_profit_op
from logic.src.policies.helpers.operators.perturbation_shaking import perturb as perturb_op
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion as greedy_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion_with_blinks as greedy_insertion_with_blinks_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_profit_insertion as greedy_profit_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_profit_insertion_with_blinks as greedy_profit_insertion_with_blinks_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_2_insertion as regret_2_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_2_profit_insertion as regret_2_profit_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_k_insertion as regret_k_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_k_profit_insertion as regret_k_profit_insertion_op,
)
from logic.src.policies.helpers.reinforcement_learning.agents.td_learning import SarsaAgent
from logic.src.policies.helpers.reinforcement_learning.features.state import StateFeatureExtractor
from logic.src.utils.functions import safe_exp

from .alns_perturbation_context import ALNSPerturbationContext


class ALNSSARSASolver:
    """
    Enhanced ALNS solver with SARSA reinforcement learning for operator selection.

    Includes all destroy, repair, perturbation, and unstringing operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        rl_params: Any,
        mandatory_nodes: Optional[List[int]] = None,
        evaluator=None,
    ):
        """Initialize enhanced ALNS-SARSA solver."""
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.evaluator = evaluator
        self.epsilon_decay_step = rl_params.sarsa_epsilon_decay_step
        self.mandatory_nodes = mandatory_nodes or []
        self.expand_pool = getattr(rl_params, "vrpp", False)
        self.profit_aware_operators = getattr(rl_params, "profit_aware_operators", False)
        self.improvement_thresholds = rl_params.sarsa_improvement_thresholds
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Initialize all operator pools
        self._init_destroy_operators()
        self._init_repair_operators()
        self._init_perturbation_operators()

        # Initialize RL components
        self.feature_extractor = StateFeatureExtractor(
            progress_thresholds=rl_params.sarsa_operator_progress_thresholds,
            stagnation_thresholds=rl_params.sarsa_operator_stagnation_thresholds,
            diversity_thresholds=rl_params.sarsa_operator_diversity_thresholds,
        )
        self.n_destroy = len(self.destroy_ops)
        self.n_repair = len(self.repair_ops)
        self.agent = SarsaAgent(
            n_states=27,  # 3*3*3 discretized states
            n_actions=self.n_destroy * self.n_repair,
            alpha=rl_params.sarsa_alpha,
            gamma=rl_params.sarsa_gamma,
            epsilon=rl_params.sarsa_epsilon,
            epsilon_decay=rl_params.sarsa_epsilon_decay,
            epsilon_min=rl_params.sarsa_epsilon_min,
            seed=params.seed,
        )
        # Seeds for agent
        self.agent_rng = np.random.default_rng(params.seed)

        # Tracking
        self.stagnation_count = 0
        self.diversity_history: Deque[float] = deque(maxlen=rl_params.sarsa_diversity_size)

    def _init_destroy_operators(self):
        """Initialize all destroy operators."""
        self.destroy_ops = [
            # Standard operators
            self._destroy_random,
            self._destroy_worst,
            self._destroy_cluster,
            # Advanced operators
            self._destroy_shaw,
            self._destroy_string,
            # Unstringing operators
            self._unstring_type_i,
            self._unstring_type_ii,
            self._unstring_type_iii,
            self._unstring_type_iv,
        ]

        self.destroy_names = [
            "Random",
            "Worst",
            "Cluster",
            "Shaw",
            "String",
            "Unstring-I",
            "Unstring-II",
            "Unstring-III",
            "Unstring-IV",
        ]

    def _init_repair_operators(self):
        """Initialize all repair operators."""
        self.repair_ops = [
            # Standard operators
            self._repair_greedy,
            self._repair_regret2,
            # Advanced operators
            lambda routes, removed: self._repair_regretk(routes, removed, k=3),
            lambda routes, removed: self._repair_regretk(routes, removed, k=4),
            self._repair_greedy_blink,
            # Stringing operators
            self._repair_string_type_i,
            self._repair_string_type_ii,
            self._repair_string_type_iii,
            self._repair_string_type_iv,
        ]

        self.repair_names = [
            "Greedy",
            "Regret-2",
            "Regret-3",
            "Regret-4",
            "Greedy-Blink",
            "String-I",
            "String-II",
            "String-III",
            "String-IV",
        ]

    def _init_perturbation_operators(self):
        """Initialize all perturbation operators."""
        self.perturbation_ops = [
            self._perturb_kick,
            self._perturb_random,
        ]

        self.perturbation_names = ["Kick", "Random"]

    # ===== Destroy Operators =====

    def _destroy_random(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Random removal."""
        return random_removal_op(routes, n, rng=self.random)

    def _destroy_worst(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Worst removal (highest cost nodes)."""
        if self.profit_aware_operators:
            return worst_profit_removal_op(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
        return worst_removal_op(routes, n, self.dist_matrix)

    def _destroy_cluster(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Cluster removal (spatially related nodes)."""
        return cluster_removal_op(routes, n, self.dist_matrix, self.nodes, rng=self.random)

    def _destroy_shaw(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Shaw removal (multi-criteria relatedness)."""
        if self.profit_aware_operators:
            return shaw_profit_removal_op(routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random)
        return shaw_removal_op(
            routes,
            n,
            self.dist_matrix,
            wastes=self.wastes,
            rng=self.random,
        )

    def _destroy_string(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """String removal (contiguous segments)."""
        return string_removal_op(routes, n, self.dist_matrix, rng=self.random)

    # ===== Repair Operators =====

    def _repair_greedy(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Greedy insertion."""
        if self.profit_aware_operators:
            return greedy_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return greedy_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.expand_pool,
        )

    def _repair_regret2(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Regret-2 insertion."""
        if self.profit_aware_operators:
            return regret_2_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return regret_2_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.expand_pool,
        )

    def _repair_regretk(self, routes: List[List[int]], removed: List[int], k: int) -> List[List[int]]:
        """Regret-k insertion (extended from regret-2)."""
        if self.profit_aware_operators:
            return regret_k_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                k=k,
                expand_pool=self.expand_pool,
            )
        return regret_k_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            k=k,
            expand_pool=self.expand_pool,
        )

    def _repair_greedy_blink(
        self, routes: List[List[int]], removed: List[int], blink_rate: float = 0.1
    ) -> List[List[int]]:
        """Greedy insertion with randomized blinks."""
        if self.profit_aware_operators:
            return greedy_profit_insertion_with_blinks_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                blink_rate=blink_rate,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
                expand_pool=self.expand_pool,
            )
        return greedy_insertion_with_blinks_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            blink_rate=blink_rate,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_i(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-I stringing repair."""
        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                1,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            1,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_ii(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-II stringing repair."""
        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                2,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_iii(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-III stringing repair."""
        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                3,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_iv(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-IV stringing repair."""
        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                4,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            4,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    # ===== Unstringing Operators =====

    def _unstring_wrapper(self, routes: List[List[int]], n: int, op_type: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper to apply unstringing moves globally as a destroy operator."""
        if self.profit_aware_operators:
            return unstringing_profit_removal(
                routes,
                n,
                op_type,
                self.dist_matrix,
                self.wastes,
                self.R,
                self.C,
                rng=self.random,
            )
        return unstringing_removal(routes, n, op_type, self.dist_matrix, rng=self.random)

    def _unstring_type_i(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-I unstringing."""
        return self._unstring_wrapper(routes, n, 1)

    def _unstring_type_ii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-II unstringing."""
        return self._unstring_wrapper(routes, n, 2)

    def _unstring_type_iii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-III unstringing."""
        return self._unstring_wrapper(routes, n, 3)

    def _unstring_type_iv(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-IV unstringing."""
        return self._unstring_wrapper(routes, n, 4)

    # ===== Perturbation Operators =====

    def _perturb_kick(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: removes random nodes and reinserts them greedily."""
        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        if self.profit_aware_operators:
            kick_profit_op(ctx, destroy_ratio=0.2, bias=2.0, rng=self.random)
        else:
            kick_op(ctx, destroy_ratio=0.2, rng=self.random)
        return ctx.routes

    def _perturb_random(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: performs random swaps to escape local optima."""
        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        perturb_op(
            ctx,
            k=self.params.perturb_k if hasattr(self.params, "perturb_k") else 3,
            prob_unvisited=self.params.prob_unvisited if hasattr(self.params, "prob_unvisited") else 0.0,
            rng=self.random,
        )
        return ctx.routes

    # ===== Main Solve Loop =====

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:  # noqa: C901
        """
        Run the enhanced ALNS search with SARSA operator selection.

        The solver iteratively applies destroy-repair operators chosen by a SARSA
        agent, adapting its strategy based on the search state (progress,
        stagnation, and diversity).

        Args:
            initial_solution (Optional[List[List[int]]]): Starting solution.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, best_profit, best_cost).
        """
        # Step 1: Initialize solution and search parameters
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit

        current_eval_profit = self.evaluator(current_routes) if self.evaluator else current_profit

        # Initialize Simulated Annealing temperature
        T = self.params.start_temp
        start_time = time.process_time()

        # Step 2: SARSA Initialization - Start with an initial state
        state_tuple = self.feature_extractor.discretize_state(0, self.params.max_iterations, 0, 1.0)
        state = self.feature_extractor.state_to_index(state_tuple)

        # Main ALNS Evolutionary loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Calculate current solution diversity to inform the state representation
            diversity = self._calculate_diversity(current_routes)
            self.diversity_history.append(diversity)

            # Step 3: Handle stagnation - If search is stuck, apply a strong perturbation (kick)
            if self.stagnation_count >= self.feature_extractor.stagnation_thresholds[1]:
                p_idx = self.random.randint(0, len(self.perturbation_ops) - 1)
                current_routes = self.perturbation_ops[p_idx](current_routes)
                current_cost = self.calculate_cost(current_routes)
                current_rev = sum(self.wastes.get(n, 0) * self.R for r in current_routes for n in r)
                current_profit = current_rev - current_cost
                current_eval_profit = self.evaluator(current_routes) if self.evaluator else current_profit
                self.stagnation_count = 0

                # Re-discretize state after escaping local optimum via perturbation
                state_tuple = self.feature_extractor.discretize_state(
                    iteration, self.params.max_iterations, self.stagnation_count, diversity
                )
                state = self.feature_extractor.state_to_index(state_tuple)

                # Record global best if perturbation accidentally found a better solution
                if current_profit > best_profit:
                    best_profit = current_profit
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = current_cost

            # Step 4: SARSA - Select action (operator pair) from current state
            # SARSA (State-Action-Reward-State-Action) is an on-policy TD control method.
            action = self.agent.select_action(state, self.agent_rng)
            d_idx = action // self.n_repair
            r_idx = action % self.n_repair

            # Step 5: Applied ALNS Core - Perform actual solution modification
            new_routes, removed = self.destroy_ops[d_idx](
                copy.deepcopy(current_routes), self._calc_removal_size(current_routes)
            )
            new_routes = self.repair_ops[r_idx](new_routes, removed)

            # Evaluate the new candidate solution
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(n, 0) * self.R for r in new_routes for n in r)
            new_profit = new_rev - new_cost

            new_eval_profit = self.evaluator(new_routes) if self.evaluator else new_profit

            # Step 6: Acceptance Criterion - decide whether to keep the new solution
            # Uses Simulated Annealing to allow occasional deterioration for diversification.
            delta = current_eval_profit - new_eval_profit
            accept = False
            if delta < self.improvement_thresholds[0]:
                accept = True
                score = 3.0  # High score for global/local improvement
            elif T > 0 and self.random.random() < safe_exp(-delta / T):
                accept = True
                score = 1.0  # Medium score for accepted non-improving moves
            else:
                accept = False
                score = 0.0  # Reject the move

            # Step 7: Reward Calculation - feedback for the SARSA agent
            if new_profit > best_profit + self.improvement_thresholds[1]:
                reward = 10.0  # Global improvement bonus
                self.stagnation_count = 0
            elif new_profit > current_profit:
                reward = 5.0  # Local improvement bonus
                self.stagnation_count = 0
            elif accept:
                reward = 1.0  # Stability/Diversification bonus
                self.stagnation_count += 1
            else:
                reward = -1.0  # Rejection penalty
                self.stagnation_count += 1

            # Step 8: SARSA - Observe next state (S') and select next action (A')
            next_state_tuple = self.feature_extractor.discretize_state(
                iteration + 1, self.params.max_iterations, self.stagnation_count, diversity
            )
            next_state = self.feature_extractor.state_to_index(next_state_tuple)
            next_action = self.agent.select_action(next_state, self.agent_rng)

            # Step 9: SARSA - Update Q-Value based on the actual observed outcome
            # SARSA Update Rule: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
            self.agent.update(state, action, reward, next_state, False, next_action)

            # Finalize transition for the next iteration
            if accept:
                current_routes = new_routes
                current_profit = new_profit
                current_eval_profit = new_eval_profit

                if current_profit > best_profit:
                    best_profit = current_profit
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = new_cost

            # Step 10: Environmental Update - Cool temperature and decay exploration
            state = next_state
            T *= self.params.cooling_rate
            if iteration % self.epsilon_decay_step == 0:
                self.agent.decay_epsilon()
            action = next_action

            # Cool temperature
            T *= self.params.cooling_rate

            # Decay exploration
            if iteration % self.epsilon_decay_step == 0:
                self.agent.decay_epsilon()

            # Visualization
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                destroy_op=self.destroy_names[d_idx],
                repair_op=self.repair_names[r_idx],
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=T,
                accepted=int(accept),
                score=score,
                reward=reward,
                epsilon=self.agent.epsilon,
                stagnation=self.stagnation_count,
                diversity=diversity,
            )

        return best_routes, best_profit, best_cost

    # ===== Helper Methods =====

    def _initialize_solve(
        self, initial_solution: Optional[List[List[int]]]
    ) -> Tuple[List[List[int]], List[List[int]], float, float]:
        """Initialize solution and metrics."""
        current_routes = initial_solution or build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )
        best_routes = copy.deepcopy(current_routes)

        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        best_profit = best_rev - best_cost
        return current_routes, best_routes, best_profit, best_cost

    def _calc_removal_size(self, routes: List[List[int]]) -> int:
        """Calculate number of nodes to remove based on current solution size."""
        current_n_nodes = sum(len(r) for r in routes)

        if current_n_nodes == 0:
            return 0

        # Scale removal percentage with minimum of 2 nodes
        lower_bound = min(current_n_nodes, 2)
        max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
        upper_bound = max(lower_bound + 1, max_pct_remove)
        upper_bound = min(upper_bound, current_n_nodes)

        return self.random.randint(lower_bound, upper_bound)

    def _calculate_diversity(self, routes: List[List[int]]) -> Any:
        """Calculate solution diversity (simple measure based on route variation)."""
        if not routes:
            return 0.0

        sizes = [len(r) for r in routes if r]
        if not sizes:
            return 0.0

        # Coefficient of variation
        mean_size = np.mean(sizes)
        if mean_size == 0:
            return 0.0

        std_size = np.std(sizes)
        cv = std_size / mean_size

        # Normalize to [0, 1]
        return min(1.0, cv)

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C
