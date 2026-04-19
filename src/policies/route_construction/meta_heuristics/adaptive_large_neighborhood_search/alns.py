"""
Adaptive Large Neighborhood Search (ALNS) for VRPP.

This module implements ALNS following Ropke & Pisinger (2005) methodology,
adapted for the Vehicle Routing Problem with Profits (VRPP) as a profit
maximization problem.

Key Implementation Features:
============================

1. **Operator Weight Updates** (Section 3.4.4):
   - Segment-based learning: w_{i,j+1} = w_{i,j}(1-r) + r * (π_i / θ_i)
   - Reaction factor r controls adaptation speed (default: 0.1)
   - Weights updated every `segment_size` iterations (default: 100)

2. **Scoring Mechanism** (Section 3.3):
   - σ₁: New global best solution found (default: 33)
   - σ₂: Better solution not visited before (default: 9)
   - σ₃: Accepted worse solution not visited before (default: 13)
   - Hash table tracks visited solutions to prevent rewarding revisits

3. **Simulated Annealing Acceptance**:
   - For VRPP profit maximization: P(accept) = exp(-Δ/T) where Δ = current_profit - new_profit.
   - This inversion (current - new) correctly handles profit maximization where
     solutions with Δ <= 0 (new >= current) are always accepted.

4. **Randomized Worst Removal** (Section 3.4.1):
   - Index selection: floor(y^p * |L|) where y ~ U(0,1) and p >= 1
   - Higher p values bias toward deterministically worst nodes
   - Default p = 3.0 provides good exploration/exploitation balance

5. **Adaptive Noise in Repair** (Section 3.4.3 / Configuration 15):
   - Applied additively to final insertion cost: C' = max{0, C + noise}
   - Noise ~ U[-η * max_dist, η * max_dist] where η (default: 0.025)
   - Clean and noisy variants are **separate operator slots** in the weight
     mechanism (paper Conf. 15), so the algorithm learns which is more
     effective rather than using a fixed 50% coin flip.
   - Operator naming: "<Name>Clean" / "<Name>Noisy"

6. **Operator Registry** (Section 3.3):
   - Destroy: random, worst, shaw [+ string, cluster, neighbor if extended_operators=True]
   - Repair: greedy × {clean,noisy}, regret-k × {clean,noisy} for k in regret_pool
   - Roulette-wheel selection (Eq. 20): p_j = w_j / Σ w_i

7. **Profit-Aware Operators** (Novel Contribution):
   - Speculative Seeding: seed_hurdle = -0.5 * detour_cost
     Allows initially unprofitable routes that may become profitable
   - Profit-based removal: targets nodes with lowest marginal profit
   - Enable via `profit_aware_operators=True` for ablation studies

Performance Optimizations:
==========================
- Shallow copy via list comprehension (replaces deepcopy): ~10x faster
- Visited solutions tracked via canonical tuple hashing
- Incremental route evaluation during operator application
- _viz_record pre-assigned as no-op when recorder=None (avoids getattr
  in hot loop)

Usage for Ablation Studies:
============================
Toggle `profit_aware_operators` in ALNSParams to compare:
- Standard ALNS (False): cost-minimization operators
- Profit-aware ALNS (True): revenue-cost maximization operators

Toggle `extended_operators` in ALNSParams to compare:
- Core destroy set (False): random, worst, shaw  [3 operators]
- Extended destroy set (True): + string, cluster, neighbor  [6 operators]

References:
===========
[1] Ropke, S., & Pisinger, D. (2006). "An adaptive large neighborhood search
    heuristic for the pickup and delivery problem with time windows."
    Transportation Science, 40(4), 455-472.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    cluster_profit_removal,
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    neighbor_profit_removal,
    neighbor_removal,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_profit_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .params import ALNSParams


class ALNSSolver:
    """
    Custom implementation of Adaptive Large Neighborhood Search for VRPP.

    Follows Ropke & Pisinger (2005) with segment-based weight updates,
    clean/noisy operator pairs (paper Configuration 15), and optional
    extended destroy operator registry.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the ALNS solver.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.vrpp = getattr(params, "vrpp", True)
        self.profit_aware_operators = getattr(params, "profit_aware_operators", False)
        self.extended_operators = getattr(params, "extended_operators", False)
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.np_random = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

        # Pre-assign no-op to avoid getattr in the hot iteration loop.
        self._viz_record: Callable = recorder.record if recorder is not None else lambda **k: None

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # ------------------------------------------------------------------
        # Destroy operator registry
        # ------------------------------------------------------------------
        self.destroy_ops, self.destroy_names = self._build_destroy_ops()

        # ------------------------------------------------------------------
        # Repair operator registry — Section 3.6 / Conf. 15:
        # Clean and noisy variants are separate operator slots so that the
        # adaptive weight mechanism can learn which is more effective.
        # Slots are interleaved: [Op0_clean, Op0_noisy, Op1_clean, Op1_noisy, ...]
        # ------------------------------------------------------------------
        self.repair_ops, self.repair_names, self.repair_noisy = self._build_repair_ops()

        # ------------------------------------------------------------------
        # Segment-based weight update (Ropke & Pisinger 2005, Section 3.4.4)
        # ------------------------------------------------------------------
        self.segment_size = params.segment_size
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)
        self.repair_counts = [0] * len(self.repair_ops)

        # Weight decay factor: (1 - r) where r is the reaction factor
        self.weight_decay = 1.0 - params.reaction_factor
        self.weight_learning_rate = params.reaction_factor

        # Visited solutions tracking (Ropke & Pisinger 2005, Section 3.3)
        self.visited_solutions: Set[Tuple[int, ...]] = set()

        # Acceptance Criterion Injection
        self.acceptance_criterion = params.acceptance_criterion
        if self.acceptance_criterion is None:
            # Fallback (should not happen if from_config is used)
            from logic.src.policies.route_construction.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="boltzmann",
                initial_temp=params.start_temp or 100.0,  # Unified fallback
                alpha=params.cooling_rate,
                seed=params.seed,
            )

    # ------------------------------------------------------------------
    # Operator registry builders
    # ------------------------------------------------------------------

    def _build_destroy_ops(self) -> Tuple[List[Callable], List[str]]:
        """
        Build the destroy operator registry.

        Core set (always included):
            random, worst[_profit], shaw[_profit]

        Extended set (when extended_operators=True):
            + string[_profit], cluster[_profit], neighbor[_profit]

        Returns:
            Tuple of (operator callables, operator names).
        """
        params = self.params
        pa = self.profit_aware_operators

        if pa:
            core: List[Tuple[str, Callable]] = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstProfitRemoval",
                    lambda r, n: worst_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        self.wastes,
                        self.R,
                        self.C,
                        p=params.worst_removal_randomness,
                        rng=self.random,  # type: ignore[arg-type]
                    ),
                ),
                (
                    "ShawProfitRemoval",
                    lambda r, n: shaw_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended: List[Tuple[str, Callable]] = [
                (
                    "StringProfitRemoval",
                    lambda r, n: string_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "ClusterProfitRemoval",
                    lambda r, n: cluster_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "NeighborProfitRemoval",
                    lambda r, n: neighbor_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
            ]
        else:
            core = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstRemoval",
                    lambda r, n: worst_removal(
                        r, n, self.dist_matrix, p=params.worst_removal_randomness, rng=self.np_random
                    ),
                ),
                (
                    "ShawRemoval",
                    lambda r, n: shaw_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended = [
                (
                    "StringRemoval",
                    lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
                ),
                (
                    "ClusterRemoval",
                    lambda r, n: cluster_removal(r, n, self.dist_matrix, nodes=self.nodes, rng=self.random),
                ),
                (
                    "NeighborRemoval",
                    lambda r, n: neighbor_removal(r, n, self.dist_matrix, rng=self.random),
                ),
            ]

        registry = core + extended if self.extended_operators else core
        ops = [fn for _, fn in registry]
        names = [nm for nm, _ in registry]
        return ops, names

    def _build_repair_ops(self) -> Tuple[List[Callable], List[str], List[bool]]:
        """
        Build the repair operator registry with clean/noisy operator pairs.

        Noise-capable operators (greedy, regret-k) produce two consecutive slots:
            [<Name>Clean, <Name>Noisy]
        The adaptive weight mechanism tracks them independently, learning
        whether noisy or clean variants perform better (paper Conf. 15).

        Operators that do not accept a noise parameter (e.g. savings) produce
        a single slot only — including them in both slots would be redundant.

        Operator set is controlled by params.regret_pool:
            "regret2"   → greedy, regret-2                        (4 slots)
            "regret234" → greedy, regret-2, regret-3, regret-4    (8 slots, default)
            "regretAll" → greedy, regret-2, regret-3, regret-4,
                          savings                                  (9 slots)

        Returns:
            Tuple of (operator callables, operator names, is_noisy flags).
        """
        params = self.params
        pa = self.profit_aware_operators
        regret_pool: str = getattr(params, "regret_pool", "regret234")

        # ------------------------------------------------------------------
        # Define the logical repair base functions.
        # Each entry: (label, fn, supports_noise)
        #   supports_noise=True  → expanded into (Clean, Noisy) slot pair
        #   supports_noise=False → single slot only
        # ------------------------------------------------------------------
        common_kw: Dict = dict(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.vrpp,
        )
        profit_kw: Dict = dict(**common_kw, R=self.R, C=self.C)

        if pa:
            base_repair: List[Tuple[str, Callable, bool]] = [
                (
                    "GreedyProfit",
                    lambda r, n, _noise=0.0: greedy_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
                (
                    "Regret2Profit",
                    lambda r, n, _noise=0.0: regret_2_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3Profit",
                        lambda r, n, _noise=0.0: regret_3_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                    (
                        "Regret4Profit",
                        lambda r, n, _noise=0.0: regret_4_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "SavingsProfit",
                        lambda r, n, _noise=0.0: savings_profit_insertion(r, n, **profit_kw),
                        False,  # savings_profit_insertion has no noise parameter; _noise ignored
                    ),
                ]
        else:
            base_repair = [
                (
                    "Greedy",
                    lambda r, n, _noise=0.0: greedy_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
                (
                    "Regret2",
                    lambda r, n, _noise=0.0: regret_2_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3",
                        lambda r, n, _noise=0.0: regret_3_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                    (
                        "Regret4",
                        lambda r, n, _noise=0.0: regret_4_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "Savings",
                        lambda r, n, _noise=0.0: savings_insertion(r, n, **common_kw),
                        False,  # savings_insertion has no noise parameter; _noise ignored
                    ),
                ]

        # ------------------------------------------------------------------
        # Expand each base heuristic:
        #   supports_noise=True  → [<Name>Clean (even idx), <Name>Noisy (odd idx)]
        #   supports_noise=False → [<Name>] (single slot, always clean)
        #
        # All slots are called as repair_op(routes, removed, noise_val) from
        # _select_and_apply_operators, so every lambda accepts a 3rd positional
        # _noise argument (even if it is discarded for clean / no-noise ops).
        # ------------------------------------------------------------------
        ops: List[Callable] = []
        names: List[str] = []
        noisy_flags: List[bool] = []

        for label, fn, supports_noise in base_repair:
            if supports_noise:
                # Clean slot: always passes 0.0; the _noise arg (from the call site) is discarded.
                # Using positional args avoids any keyword-name mismatch across base lambdas.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, 0.0))
                names.append(f"{label}Clean")
                noisy_flags.append(False)
                # Noisy slot: forwards _noise (the noise value computed by _get_noise) positionally.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, _noise))
                names.append(f"{label}Noisy")
                noisy_flags.append(True)
            else:
                # Single slot: no noise expansion; fn already accepts and discards _noise=0.0.
                ops.append(fn)
                names.append(label)
                noisy_flags.append(False)

        return ops, names, noisy_flags

    # ------------------------------------------------------------------
    # Core solver loop helpers
    # ------------------------------------------------------------------

    def _get_noise(self) -> float:
        """
        Generate noise for repair operators (Ropke & Pisinger 2005, Section 3.4.3).

        Noise is scaled relative to the maximum distance in the problem instance:
        noise ~ U[-eta * max_dist, eta * max_dist]

        Returns:
            float: Noise value in range [-eta * max_dist, eta * max_dist]
        """
        max_dist = self.dist_matrix.max()
        return self.random.uniform(-self.params.noise_factor, self.params.noise_factor) * max_dist

    def _hash_solution(self, routes: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """
        Create a canonical hash of a solution for tracking visited states.

        Returns:
            Tuple representation of sorted routes for hashing.
        """
        sorted_routes = sorted([tuple(route) for route in routes if route])
        return tuple(sorted_routes)

    def _initialize_solve(self, initial_solution: Optional[List[List[int]]]):
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = [r[:] for r in current_routes]
        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        best_profit = best_rev - best_cost
        return current_routes, best_routes, best_profit, best_cost

    def _select_and_apply_operators(self, current_routes: List[List[int]]) -> Tuple[List[List[int]], int, int]:
        """
        Select and apply one destroy + one repair operator.

        Repair operators are selected via roulette-wheel (Eq. 20). Clean and
        noisy variants are independent slots — the adaptive mechanism decides
        which is preferable rather than a fixed coin flip (paper Conf. 15).

        Returns:
            (new_routes, destroy_idx, repair_idx)
        """
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)
        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        current_n_nodes = sum(len(route) for route in current_routes)
        if current_n_nodes == 0:
            n_remove = 0
        else:
            lower_bound = min(current_n_nodes, self.params.min_removal)
            max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
            # Upper bound: min(100, ξ * n) per paper §4.3.1
            upper_bound = min(current_n_nodes, self.params.max_removal_cap, max(lower_bound, max_pct_remove))
            upper_bound = max(upper_bound, lower_bound)
            n_remove = self.random.randint(lower_bound, upper_bound)

        # Noise value is only computed when a noisy repair slot was selected.
        # The adaptive weight mechanism controls how often noisy slots are chosen.
        noise_val = self._get_noise() if self.repair_noisy[r_idx] else 0.0

        new_routes, removed = destroy_op([route[:] for route in current_routes], n_remove)
        new_routes = repair_op(new_routes, removed, noise_val)
        return new_routes, d_idx, r_idx

    # Removed _accept_solution in favor of self.acceptance_criterion.accept()

    def _update_weights(self, d_idx: int, r_idx: int, score: float) -> None:
        self.destroy_scores[d_idx] += score
        self.repair_scores[r_idx] += score
        self.destroy_counts[d_idx] += 1
        self.repair_counts[r_idx] += 1

    def _end_segment(self) -> None:
        """
        Update operator weights at the end of a segment (Ropke & Pisinger 2005, Eq. 4.1).

        Formula: w_{i,j+1} = w_{i,j} * (1-r) + r * (π_i / θ_i)
        where:
            - w_{i,j} is the weight of operator i in segment j
            - r is the reaction factor (learning rate)
            - π_i is the total score accumulated by operator i
            - θ_i is the number of times operator i was used
        """
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = (
                    self.weight_decay * self.destroy_weights[i] + self.weight_learning_rate * avg_score
                )
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0

        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    self.weight_decay * self.repair_weights[i] + self.weight_learning_rate * avg_score
                )
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0

    # Removed _calculate_dynamic_start_temp in favor of criterion.setup()

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        start_time = time.process_time()
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit

        self.visited_solutions.clear()
        self.visited_solutions.add(self._hash_solution(best_routes))  # type: ignore[arg-type]

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.setup(current_profit)

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            new_routes, d_idx, r_idx = self._select_and_apply_operators(current_routes)
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost

            new_hash = self._hash_solution(new_routes)
            is_new_solution = new_hash not in self.visited_solutions

            if self.acceptance_criterion is None:
                accept = new_profit > current_profit + 1e-6
            else:
                accept, _ = self.acceptance_criterion.accept(
                    current_obj=current_profit,
                    candidate_obj=new_profit,
                    f_best=best_profit,
                    iteration=_it,
                    max_iterations=self.params.max_iterations,
                )

            # ------------------------------------------------------------------
            # Scoring (Ropke & Pisinger 2005, Section 3.4):
            #   σ₁ — new global best (no "not visited" qualifier)
            #   σ₂ — better than current, not visited before, accepted
            #   σ₃ — worse than current, not visited before, accepted
            # ------------------------------------------------------------------
            score = 0.0
            if new_profit > best_profit + 1e-6:
                best_routes = [r[:] for r in new_routes]
                best_profit = new_profit
                best_cost = new_cost
                score = self.params.sigma_1
            elif accept and is_new_solution:
                score = self.params.sigma_2 if new_profit > current_profit + 1e-6 else self.params.sigma_3

            if accept:
                self.visited_solutions.add(new_hash)  # type: ignore[arg-type]
                current_routes = new_routes
                current_profit = new_profit

            self._update_weights(d_idx, r_idx, score)
            if (_it + 1) % self.segment_size == 0:
                self._end_segment()

            if self.acceptance_criterion is not None:
                self.acceptance_criterion.step(
                    current_obj=current_profit,
                    candidate_obj=new_profit,
                    accepted=accept,
                    f_best=best_profit,
                    iteration=_it,
                    max_iterations=self.params.max_iterations,
                )

            self._viz_record(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=best_profit,
                current_profit=current_profit,
                acceptance_state=self.acceptance_criterion.get_state() if self.acceptance_criterion else {},
                accepted=int(accept),
                score=score,
            )
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def select_operator(self, weights: List[float]) -> int:
        """Roulette-wheel operator selection (Eq. 20)."""
        total = sum(weights)
        r = self.random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
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

    def build_initial_solution(self) -> List[List[int]]:
        """Build an initial solution using the greedy profit-aware heuristic."""
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )
