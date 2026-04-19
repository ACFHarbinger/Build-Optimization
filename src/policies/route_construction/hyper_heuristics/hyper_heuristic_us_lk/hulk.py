"""
HULK (Hyper-heuristic Using unstringing/stringing with Local search and K-opt).

A hyper-heuristic that combines:
- Unstringing operators for solution destruction
- Stringing operators for solution reconstruction
- Local search operators for improvement
- Adaptive operator selection based on performance

Reference:
    Müller, L. F., & Bonilha, I. (2022). "Hyper-Heuristic Based on ACO
    and Local Search for Dynamic Optimization Problems."
    Algorithms, 15(1), 9. https://doi.org/10.3390/a15010009
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes

from .adaptive_selection import AdaptiveOperatorSelector
from .operators import HULKOperators
from .params import HULKParams
from .solution import Solution


class HULKSolver:
    """
    HULK: Hyper-heuristic Using unstringing/stringing with Local search and K-opt.

    Main solver class that coordinates operator selection and solution improvement.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HULKParams,
        mandatory_nodes: Optional[List[int]] = None,
        evaluator=None,
    ):
        """
        Initialize HULK solver.

        Args:
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: HULK parameters.
            mandatory_nodes: Must-visit nodes.
            evaluator: Optional custom evaluation function.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.evaluator = evaluator
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1

        # Initialize operators
        self.ops = HULKOperators(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            mandatory_nodes,
            params.seed,
            expand_pool=getattr(self.params, "vrpp", False),
            profit_aware_operators=getattr(self.params, "profit_aware_operators", False),
        )

        # Initialize adaptive selectors for different operator types
        self.unstring_selector = AdaptiveOperatorSelector(
            operators=params.unstring_operators,
            epsilon=params.epsilon,
            memory_size=params.memory_size,
            learning_rate=params.weight_learning_rate,
            weight_decay=params.weight_decay,
            seed=params.seed,
        )

        self.string_selector = AdaptiveOperatorSelector(
            operators=params.string_operators,
            epsilon=params.epsilon,
            memory_size=params.memory_size,
            learning_rate=params.weight_learning_rate,
            weight_decay=params.weight_decay,
            seed=params.seed,
        )

        self.local_search_selector = AdaptiveOperatorSelector(
            operators=params.local_search_operators,
            epsilon=params.epsilon,
            memory_size=params.memory_size,
            learning_rate=params.weight_learning_rate,
            weight_decay=params.weight_decay,
            seed=params.seed,
        )

        # Simulated annealing temperature
        self.temperature = params.start_temp

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run HULK hyper-heuristic search.

        Args:
            initial_solution: Optional starting solution.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        global_best_solution = None
        global_best_profit = float("-inf")
        global_best_cost = 0.0

        for restart in range(self.params.restarts):
            # Initialize solution
            if initial_solution and restart == 0:
                routes = [list(r) for r in initial_solution]
            else:
                routes = build_greedy_routes(
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    rng=self.rng,
                )

            current = Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

            restart_best = current.copy()
            restart_best_profit = restart_best.profit or 0.0

            # Reset temperature
            self.temperature = self.params.start_temp

            start_time = time.process_time()
            it = 0
            last_improvement_it = 0

            while it < self.params.max_iterations:
                # Check time limit
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                it += 1

                # Apply unstring-string-local search cycle
                neighbor, unstring_op, string_op, ls_op = self._apply_operators(current)

                # Acceptance decision
                accepted, delta = self._accept_solution(current, neighbor)

                # Müller & Bonilha (2022) - Enhanced Reward Structure (Alpha, Beta, Gamma, Delta)
                # Alpha: New global best
                # Beta: Improvement over current solution
                # Gamma: Accepted worsening move
                # Delta: Rejected move

                is_global_best = (neighbor.profit or 0.0) > global_best_profit
                is_improvement = (neighbor.profit or 0.0) > (current.profit or 0.0)

                if is_global_best:
                    reward = self.params.score_alpha
                elif is_improvement:
                    reward = self.params.score_beta
                elif accepted:
                    reward = self.params.score_gamma
                else:
                    reward = self.params.score_delta

                # Update selectors
                self.unstring_selector.update(unstring_op, reward, 0.01, is_global_best)
                self.string_selector.update(string_op, reward, 0.01, is_global_best)
                if ls_op:
                    self.local_search_selector.update(ls_op, reward, 0.01, is_global_best)

                if accepted:
                    current = neighbor

                    if (current.profit or 0.0) > restart_best_profit:
                        restart_best = current.copy()
                        restart_best_profit = current.profit or 0.0
                        last_improvement_it = it

                # Cool temperature
                self.temperature *= self.params.cooling_rate
                self.temperature = max(self.params.min_temp, self.temperature)

                # Decay exploration
                if it % 10 == 0:
                    self.unstring_selector.decay_epsilon(self.params.epsilon_decay, self.params.min_epsilon)
                    self.string_selector.decay_epsilon(self.params.epsilon_decay, self.params.min_epsilon)
                    self.local_search_selector.decay_epsilon(self.params.epsilon_decay, self.params.min_epsilon)

                # Visualization
                getattr(self, "_viz_record", lambda **k: None)(
                    iteration=it + restart * self.params.max_iterations,
                    unstring_op=unstring_op,
                    string_op=string_op,
                    local_search_op=ls_op or "None",
                    best_profit=restart_best_profit,
                    current_profit=current.profit,
                    temperature=self.temperature,
                    accepted=int(accepted),
                    delta=delta,
                )

                # Restart if stuck
                if it - last_improvement_it > self.params.restart_threshold:
                    break

            # Update global best
            if restart_best_profit > global_best_profit:
                global_best_solution = restart_best
                global_best_profit = restart_best_profit
                global_best_cost = restart_best.cost or 0.0

        return (
            global_best_solution.routes if global_best_solution else [[]],
            global_best_profit,
            global_best_cost,
        )

    def _apply_operators(self, solution: Solution) -> Tuple[Solution, str, str, Optional[str]]:
        """
        Apply the full operator cycle: unstring -> string -> local search (LKS/K-opt).

        Follows Müller & Bonilha (2022) three-phase cycle:
        1. Destruction (Unstringing)
        2. Reconstruction (Stringing)
        3. Improvement (Local Search / K-opt)

        Returns:
            (new_solution, unstring_op_name, string_op_name, ls_op_name)
        """
        # 1. Select and apply unstringing (destroy)
        unstring_op = self.unstring_selector.select_operator()
        n_remove = self._calculate_removal_size(solution)

        unstring_funcs = {
            "type_i": self.ops.apply_unstring_type_i,
            "type_ii": self.ops.apply_unstring_type_ii,
            "type_iii": self.ops.apply_unstring_type_iii,
            "type_iv": self.ops.apply_unstring_type_iv,
            "shaw": self.ops.apply_unstring_shaw,
            "string": self.ops.apply_unstring_string,
        }

        destroyed, removed = unstring_funcs[unstring_op](solution, n_remove)

        # 2. Select and apply stringing (repair)
        string_op = self.string_selector.select_operator()
        repaired = self.ops.apply_string_repair(destroyed, removed, string_op)

        # 3. Apply Local Search (LKS / K-opt)
        # In HULK, LS is applied to intensified areas or as a general improvement step
        ls_op = self.local_search_selector.select_operator()

        # Consistent with paper, we apply LS with a certain probability or on best
        if self.rng.random() < 0.5:
            ls_funcs = {
                "2-opt": self.ops.apply_2_opt,
                "3-opt": self.ops.apply_3_opt,
                "swap": self.ops.apply_swap,
                "relocate": self.ops.apply_relocate,
            }

            if ls_op in ls_funcs:
                improved = ls_funcs[ls_op](repaired)
                return improved, unstring_op, string_op, ls_op

        return repaired, unstring_op, string_op, ls_op

    def _calculate_removal_size(self, solution: Solution) -> int:
        """Calculate number of nodes to remove."""
        total_nodes = solution.get_total_nodes()
        if total_nodes == 0:
            return 0

        min_remove = self.params.min_destroy_size
        max_remove = max(min_remove + 1, int(total_nodes * self.params.max_destroy_pct))
        max_remove = min(max_remove, total_nodes)

        return self.rng.randint(min_remove, max_remove)

    def _accept_solution(self, current: Solution, neighbor: Solution) -> Tuple[bool, float]:
        """
        Decide whether to accept neighbor solution.

        Uses simulated annealing acceptance criterion.

        Returns:
            (accepted, delta_profit)
        """
        delta = (neighbor.profit or 0.0) - (current.profit or 0.0)

        if delta > 0:
            return True, delta

        # Simulated annealing acceptance
        if self.temperature > 0:
            prob = math.exp(delta / self.temperature)
            if self.rng.random() < prob:
                return True, delta

        # Random acceptance with small probability
        if self.rng.random() < self.params.accept_worse_prob:
            return True, delta

        return False, delta

    def get_operator_statistics(self) -> Dict:
        """Get performance statistics for all operators."""
        return {
            "unstring": self.unstring_selector.get_statistics(),
            "string": self.string_selector.get_statistics(),
            "local_search": self.local_search_selector.get_statistics(),
        }
