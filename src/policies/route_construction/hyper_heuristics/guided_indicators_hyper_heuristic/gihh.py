"""
Guided Indicators Hyper-Heuristic (GIHH) for VRPP.

GIHH is an adaptive hyper-heuristic that selects Low-Level Heuristics (LLHs)
based on their historical performance, guided by two specific indicators.
Refactored to rigorously align with Chen et al. (2018):
1. ScoreA (Quality Reward): Increment on acceptance to Pareto Archive.
2. ScoreB (Directional Reward): Track objective optimization bias (Revenue vs Cost).

References:
    - Chen, B., Qu, R., Bai, R., & Laesanklang, W. (2018). "A hyper-heuristic with
      two guidance indicators for bi-objective mixed-shift vehicle routing problem
      with time windows." European Journal of Operational Research, 269(2), 661-675.
"""

import copy
import random
import time
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager
from logic.src.policies.helpers.operators import (
    apply_ges,
    apply_intra_route_cross_exchange,
    apply_lns,
    build_greedy_routes,
)
from logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.indicators import (
    ScoreAIndicator,
    ScoreBIndicator,
)
from logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params import GIHHParams
from logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution import Solution


class GIHHSolver:
    """
    GIHH solver implementation using a Pareto Archive.

    This hyper-heuristic uses Episodic Weight Updates with Roulette Wheel selection
    and maintains a Multi-Objective Archive (ARCH).

    OBJECTIVE ALIGNMENT (Chen et al. 2018 vs VRPP):
      - MS-VRPTW Objective 1 (Minimize DP) -> VRPP Objective 1 (Maximize Revenue).
      - MS-VRPTW Objective 2 (Minimize TD) -> VRPP Objective 2 (Minimize Cost).
      - Note: A positive ScoreB and positive DEVIATION implies an algorithmic
        inclination toward improving Revenue (the first objective).

    For integration with WSmart-Route, it ultimately exports the archive solution
    that maximizes the scalar `profit`.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GIHHParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()

        # Initialize new guidance indicators
        self.score_a = ScoreAIndicator()
        self.score_b = ScoreBIndicator()

        # The 5 LLHs from the reference paper
        self.operators = [
            "inter_route_2opt_star",
            "inter_route_cross_exchange",
            "intra_route_cross_exchange",
            "lns",
            "ges",
        ]

        # Episodic Learning weights
        self.weights = {op: 1.0 / len(self.operators) for op in self.operators}
        self.applied_times = {op: 0 for op in self.operators}

        # Track global Best Profit for WSmart-Route output contract
        self.global_best_profit_sol: Optional[Solution] = None
        self.ARCH: List[Solution] = []
        self.segment_start_sc: Optional[Solution] = None
        self.segment_accepted_sols: List[Solution] = []

    def solve(self) -> List[Solution]:
        """Runs the GIHH metaheuristic using Pareto evaluation."""
        if len(self.dist_matrix) <= 1:
            return []

        start_time = time.process_time()

        # 1. Generate Initial Solution
        # [METHODOLOGY NOTE]
        # In the original MS-VRPTW paper (Chen et al., 2018), an Emergency Level-Based
        # Insertion Heuristic (EBIH) was employed due to the complexity of shift
        # constraints. For this generalized VRPP adaptation, `build_greedy_routes`
        # serves as the equivalent problem-specific baseline, providing a high-quality
        # starting point for hyper-volume metrics and Pareto frontier convergence.
        initial_routes = build_greedy_routes(
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )

        initial_sol = Solution(
            initial_routes,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
        )

        # Multi-Objective Archive
        self.ARCH = [initial_sol]
        self.global_best_profit_sol = copy.deepcopy(initial_sol)

        # Setup modular acceptance criterion
        assert self.params.acceptance_criterion is not None
        self.params.acceptance_criterion.setup(initial_sol.profit)

        iteration = 1
        iterations_since_last_improvement = 0

        while True:
            # Check stopping criteria
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break
            if iteration > self.params.max_iterations:
                break
            if iterations_since_last_improvement >= self.params.nonimp_threshold:
                break

            # 2. Select Current Solution from ARCH
            current_sol = self.rng.choice(self.ARCH)

            if iteration % self.params.seg == 1 or self.segment_start_sc is None:
                self.segment_start_sc = current_sol

            # 3. Select operator
            selected_operator = self._select_operator()

            # 4. Apply selected operator
            candidate_sol = self._apply_selected_operator(current_sol, selected_operator)

            # WSmart-Route compatibility: track global scalar best
            if candidate_sol.is_feasible() and candidate_sol.profit > self.global_best_profit_sol.profit:
                self.global_best_profit_sol = copy.deepcopy(candidate_sol)

            # 5. Evaluate Acceptance
            self.applied_times[selected_operator] += 1

            # Use modular criterion to decide acceptance
            assert self.params.acceptance_criterion is not None
            is_accepted, _ = self.params.acceptance_criterion.accept(
                current_obj=current_sol.profit,
                candidate_obj=candidate_sol.profit,
                iteration=iteration,
                max_iterations=self.params.max_iterations,
                archive=self.ARCH,
                candidate_solution=candidate_sol,
            )

            if is_accepted:
                # Still perform the Pareto archive update logic
                _arch_updated = self._update_archive(candidate_sol)
                self.segment_accepted_sols.append(candidate_sol)
                iterations_since_last_improvement = 0
            else:
                iterations_since_last_improvement += 1

            # Step the criterion after move selection
            assert self.params.acceptance_criterion is not None
            self.params.acceptance_criterion.step(
                current_obj=current_sol.profit,
                candidate_obj=candidate_sol.profit,
                accepted=is_accepted,
                iteration=iteration,
            )

            # 6. Update Guidance Indicators
            self.score_a.update(selected_operator, is_accepted)
            if is_accepted:
                rev_improved = candidate_sol.revenue_total > current_sol.revenue_total
                cost_improved = candidate_sol.cost < current_sol.cost
                self.score_b.update(selected_operator, rev_improved, cost_improved)

            # 7. Episodic Weight Updates
            if iteration % self.params.seg == 0:
                self._update_episodic_weights()
                self.segment_accepted_sols.clear()

            iteration += 1

        return self.ARCH

    def _select_operator(self) -> str:
        """Proportional Roulette Wheel Selection with a minimum probability guarantee."""
        min_p = self.params.min_prob
        n_ops = len(self.operators)

        # Calculate effective sum after setting aside minimums
        remaining_prob = 1.0 - (n_ops * min_p)
        total_weight = sum(self.weights.values())

        probs = []
        for op in self.operators:
            p = min_p + remaining_prob * (self.weights[op] / total_weight) if total_weight > 0 else 1.0 / n_ops
            probs.append(p)

        # Roulette wheel sample
        return self.rng.choices(self.operators, weights=probs, k=1)[0]

    def _apply_selected_operator(self, current: Solution, operator: str) -> Solution:
        """Apply the specific paper LLHs using WSmart-Route operator parity."""
        candidate = current.copy()

        if operator in ["inter_route_2opt_star", "inter_route_cross_exchange"]:
            ls = LocalSearchManager(
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                improvement_threshold=0.0,
                seed=self.params.seed,
            )
            ls.set_routes(candidate.routes)
            if operator == "inter_route_2opt_star":
                ls.two_opt_star()
            elif operator == "inter_route_cross_exchange":
                ls.cross_exchange_op()
            candidate.routes = ls.get_routes()
        elif operator == "intra_route_cross_exchange":
            candidate.routes = apply_intra_route_cross_exchange(candidate.routes, self.rng)
        elif operator == "lns":
            candidate.routes = apply_lns(
                candidate.routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.rng
            )
        elif operator == "ges":
            candidate.routes = apply_ges(candidate.routes, self.dist_matrix, self.wastes, self.capacity, self.rng)

        candidate.evaluate()
        return candidate

    def _update_archive(self, candidate: Solution) -> bool:
        """
        Updates the Pareto Archive.
        Returns True if candidate is non-dominated by any solution in ARCH.
        """
        if not candidate.is_feasible():
            return False

        is_dominated = False
        to_remove = []

        for arch_sol in self.ARCH:
            # Step 2: Fix Archive Explosion Vulnerability
            # Reject structurally identical clones to prevent the archive from
            # being flooded with duplicates that do not strictly dominate each other.
            if (
                arch_sol.profit == candidate.profit
                and arch_sol.cost == candidate.cost
                and candidate.is_identical_to(arch_sol)
            ):
                return False

            if arch_sol.dominates(candidate):
                is_dominated = True
                break
            elif candidate.dominates(arch_sol):
                to_remove.append(arch_sol)

        if not is_dominated:
            for sol in to_remove:
                self.ARCH.remove(sol)
            self.ARCH.append(candidate)
            return True

        return False

    def _update_episodic_weights(self) -> None:
        """
        Episodic Weight Update Mechanism (Chen et al. 2018).

        OBJECTIVE ALIGNMENT:
        A positive DEVIATION implies the search in this segment heavily
        favored Revenue improvements over Cost improvements.
        """
        # Phase 1: Quality Reward update
        for op in self.operators:
            a_score = self.score_a.get_score(op)
            k = self.applied_times[op]
            ratio = a_score / k if k > 0 else 0.0

            # weight_i = alpha * weight_i + beta * (ScoreA_i / applied_times_i)
            self.weights[op] = self.params.alpha * self.weights[op] + self.params.beta * ratio

        # Phase 2: Directional Reward update based on Segment Bias
        if self.segment_start_sc is not None and self.segment_accepted_sols:
            rev_improvements = sum(
                1 for s in self.segment_accepted_sols if s.revenue_total > self.segment_start_sc.revenue_total
            )
            cost_improvements = sum(1 for s in self.segment_accepted_sols if s.cost < self.segment_start_sc.cost)

            # Equation 24: Rigorous Directional Deviation Calculation
            # We replace the informal '- 0.5' denominator hack with a strict check
            # for zero improvements to ensure theoretical stability.
            total_improvements = rev_improvements + cost_improvements
            dev = (rev_improvements - cost_improvements) / total_improvements if total_improvements > 0 else 0.0
        else:
            dev = 0.0

        for op in self.operators:
            b_score = self.score_b.get_score(op)
            k = self.applied_times[op]
            b_ratio = b_score / k if k > 0 else 0.0

            # Step 1: Fix the "Equation 25 Paradox" (Directional Balancing)
            # The original Chen et al. (2018) formula (Equation 25) suggests increasing
            # weights when they align with the current search bias (dev * b_ratio > 0).
            # However, this exacerbates heuristic bias toward a single objective.
            # We implement a theoretical correction for multi-objective balancing:
            # trigger the weight update only when the operator's directional bias
            # is OPPOSITE to the segment's dominant deviation, effectively regaining
            # equilibrium in the Pareto frontier exploration.
            if (dev > 0 and b_ratio < 0) or (dev < 0 and b_ratio > 0):
                self.weights[op] -= self.params.gamma * b_ratio
                self.weights[op] = max(0.0, self.weights[op])  # Prevent negative weights

        # Reset tallies
        for op in self.operators:
            self.score_a.reset(op)
            self.score_b.reset(op)
            self.applied_times[op] = 0

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates total distance (km)."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i]][route[i + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
