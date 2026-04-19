"""
Adaptive Large Neighborhood Search with Inter-Period Operators (ALNS-IPO).

Extends the standard ``ALNSSolver`` to optimise a full T-day horizon
chromosome ``horizon_routes: List[List[List[int]]]`` by incorporating
inter-period destroy/repair operators (IPO) that can move and remove visits
across days.

Architecture
------------
``ALNSSolverIPO`` is a **subclass** of ``ALNSSolver``.  It:

1. Inherits the full operator registry (destroy/repair) and the
   segment-based weight update machinery from the parent class.
2. **Appends** inter-period operators (``ShiftVisitRemoval``,
   ``PatternRemoval``) to the destroy registry with their own weight
   slots — meaning the adaptive mechanism can learn their relative value.
3. Replaces the standard repair operator collection with the
   ``forward_looking_insertion`` repair for inter-period slots.
4. Exposes a new ``solve_horizon()`` entry point that returns the full
   T-day plan alongside Day-0 routes for the simulator.

Chromosome representation
--------------------------
The ALNS-IPO maintains a **horizon chromosome**::

    horizon_routes: List[List[List[int]]]
    # [day_index][route_index][node_index]

During each iteration, single-day operators act on the chromosome of a
randomly chosen day, while inter-period operators act on the full horizon.

References
----------
Coelho, L. C., Cordeau, J.-F., & Laporte, G. (2012).
"The inventory-routing problem with transshipment."
Computers & Operations Research, 39(11), 2537–2548.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    forward_looking_insertion,
    greedy_horizon_insertion,
    greedy_insertion,
    greedy_profit_insertion,
    pattern_removal,
    random_horizon_removal,
    regret_k_temporal_insertion,
    shaw_horizon_removal,
    shift_visit_removal,
    stochastic_aware_insertion,
    urgency_aware_removal,
    worst_profit_horizon_removal,
)
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import (
    ALNSSolver,
)
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.params import (
    ALNSIPOParams,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder


class ALNSSolverIPO(ALNSSolver):
    """ALNS with Inter-Period Operators (IPO) for T-day horizon optimization.

    Adds inter-period destroy operators (ShiftVisitRemoval, PatternRemoval)
    and the ForwardLookingInsertion repair to the adaptive weight registry.
    Single-day operators inherited from ``ALNSSolver`` are preserved and
    can be applied to individual days of the horizon.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSIPOParams,
        mandatory_nodes: Optional[List[int]] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ) -> None:
        """Initialize the multi-period ALNS solver.

        Args:
            dist_matrix: Square distance/cost matrix (n+1 × n+1, index 0 = depot).
            wastes: Mapping {node_id: fill_level} for all customer nodes.
            capacity: Vehicle capacity.
            R: Revenue per unit waste collected.
            C: Cost per unit distance.
            params: ALNS-IPO parameters including multi-period extensions.
            mandatory_nodes: Nodes that must be visited on Day 0.
            recorder: Optional policy state recorder.
        """
        super().__init__(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_nodes=mandatory_nodes,
            recorder=recorder,
        )

        self.horizon = params.horizon
        self.stockout_penalty = params.stockout_penalty
        self.forward_looking_depth = params.forward_looking_depth
        self.shift_direction = params.shift_direction
        self.inventory_lambda = params.inventory_lambda
        self.inter_period_weight_init = params.inter_period_weight
        self.inter_period_operators_enabled = params.inter_period_operators

        self.stochastic_repair = params.stochastic_repair

        # Store pool starting indices for inter-period selection
        self.ip_destroy_start = len(self.destroy_ops)
        self.ip_repair_start = len(self.repair_ops)

        # Register inter-period operators
        if self.inter_period_operators_enabled:
            self._append_inter_period_destroy_ops()
            self._append_inter_period_repair_ops()

        # Scenario tree is injected at solve_horizon() time
        self._scenario_tree: Optional[Any] = None

    # ------------------------------------------------------------------
    # Operator registry extension
    # ------------------------------------------------------------------

    def _append_inter_period_destroy_ops(self) -> None:
        """Append ShiftVisitRemoval, PatternRemoval, and new horizon operators to the destroy registry."""
        direction = self.shift_direction

        # 1. ShiftVisitRemoval
        def _shift(r: Any, n: int) -> Tuple[Any, Any]:
            return shift_visit_removal(r, n, direction=direction, wastes=self.wastes, rng=self.random)

        # 2. PatternRemoval
        def _pattern(r: Any, n: int) -> Tuple[Any, Any]:
            return pattern_removal(r, n, wastes=self.wastes, rng=self.random)

        # 3. RandomHorizonRemoval
        def _random_h(r: Any, n: int) -> Tuple[Any, Any]:
            return random_horizon_removal(r, n, rng=self.random)

        # 4. WorstProfitHorizonRemoval
        def _worst_h(r: Any, n: int) -> Tuple[Any, Any]:
            return worst_profit_horizon_removal(r, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random)

        # 5. ShawHorizonRemoval
        def _shaw_h(r: Any, n: int) -> Tuple[Any, Any]:
            return shaw_horizon_removal(r, n, self.dist_matrix, rng=self.random)

        # 6. UrgencyAwareRemoval
        def _urgency_h(r: Any, n: int) -> Tuple[Any, Any]:
            return urgency_aware_removal(r, n, self.wastes, rng=self.random)

        new_ops = [
            (_shift, "ShiftVisitRemoval"),
            (_pattern, "PatternRemoval"),
            (_random_h, "RandomHorizonRemoval"),
            (_worst_h, "WorstProfitHorizonRemoval"),
            (_shaw_h, "ShawHorizonRemoval"),
            (_urgency_h, "UrgencyAwareRemoval"),
        ]

        for op, name in new_ops:
            self.destroy_ops.append(op)
            self.destroy_names.append(name)
            self.destroy_weights.append(self.inter_period_weight_init)
            self.destroy_scores.append(0.0)
            self.destroy_counts.append(0)

    def _append_inter_period_repair_ops(self) -> None:
        """Append Horizon insertion operators to the repair registry."""

        def _greedy_h(r: Any, rem: Any, _noise: float = 0.0) -> Any:
            return greedy_horizon_insertion(
                r,
                rem,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                scenario_tree=self._scenario_tree,
                use_stochastic=self.stochastic_repair,
            )

        def _regret_h(r: Any, rem: Any, _noise: float = 0.0) -> Any:
            return regret_k_temporal_insertion(
                r,
                rem,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                scenario_tree=self._scenario_tree,
                use_stochastic=self.stochastic_repair,
            )

        def _stoch_h(r: Any, rem: Any, _noise: float = 0.0) -> Any:
            return stochastic_aware_insertion(
                r, rem, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, scenario_tree=self._scenario_tree
            )

        new_ops = [
            (_greedy_h, "GreedyHorizonInsertion"),
            (_regret_h, "RegretTemporalInsertion"),
            (_stoch_h, "StochasticAwareInsertion"),
            (self._make_fl_repair(), "ForwardLookingRepair"),
        ]

        for op, name in new_ops:
            self.repair_ops.append(op)
            self.repair_names.append(name)
            self.repair_weights.append(self.inter_period_weight_init)
            self.repair_scores.append(0.0)
            self.repair_counts.append(0)

    def _make_fl_repair(self) -> Callable:
        """Build the forward-looking repair operator closure."""

        def _fl(horizon_routes: List[List[List[int]]], removed: Any, _noise: float = 0.0) -> List[List[List[int]]]:
            return forward_looking_insertion(
                horizon_routes=horizon_routes,
                removed=removed,
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                scenario_tree=self._scenario_tree,
                stockout_penalty=self.stockout_penalty,
                look_ahead_days=self.forward_looking_depth,
                lambda_inventory=self.inventory_lambda,
            )

        return _fl

    # ------------------------------------------------------------------
    # Helper: build initial horizon solution
    # ------------------------------------------------------------------

    def build_initial_horizon_solution(self) -> List[List[List[int]]]:
        """Build an initial T-day horizon solution using the parent greedy heuristic."""
        horizon: List[List[List[int]]] = []
        for _ in range(self.horizon):
            day_routes = build_greedy_routes(
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
            )
            horizon.append(day_routes)
        return horizon

    # ------------------------------------------------------------------
    # Horizon cost evaluation
    # ------------------------------------------------------------------

    def calculate_horizon_cost(self, horizon_routes: List[List[List[int]]]) -> float:
        """Compute total routing cost across the horizon."""
        total = 0.0
        for day_routes in horizon_routes:
            total += self.calculate_cost(day_routes)
        return total

    def calculate_horizon_profit(self, horizon_routes: List[List[List[int]]]) -> float:
        """Compute total profit (revenue − routing cost) across the horizon."""
        total_cost = self.calculate_horizon_cost(horizon_routes)
        total_rev = sum(
            self.wastes.get(node, 0.0) * self.R
            for day_routes in horizon_routes
            for route in day_routes
            for node in route
        )
        return total_rev - total_cost

    # ------------------------------------------------------------------
    # Multi-period solve
    # ------------------------------------------------------------------

    def solve_horizon(
        self,
        initial_horizon_routes: Optional[List[List[List[int]]]] = None,
        scenario_tree: Optional[Any] = None,
    ) -> Tuple[List[List[List[int]]], float, float]:
        """Solve the T-day multi-period ALNS.

        Alternates between:
        - Single-day operators (applied to a randomly chosen day)
        - Inter-period operators (applied to the full horizon)

        Args:
            initial_horizon_routes: Initial T-day solution. If None, built greedily.
            scenario_tree: Optional ScenarioTree for forward-looking inventory evaluation.

        Returns:
            Tuple of:
                - best_horizon_routes: Best T-day plan found.
                - best_profit: Total horizon profit.
                - best_cost: Total horizon routing cost.
        """
        self._scenario_tree = scenario_tree
        start_time = time.process_time()

        # Initialise
        current_horizon = initial_horizon_routes or self.build_initial_horizon_solution()
        best_horizon = [[[*r] for r in day] for day in current_horizon]
        best_cost = self.calculate_horizon_cost(best_horizon)
        best_profit = self.calculate_horizon_profit(best_horizon)
        current_profit = best_profit

        self.visited_solutions.clear()

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.setup(current_profit)

        # No longer using fixed n_inter_period = 2

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            d_idx = self.select_operator(self.destroy_weights)
            is_inter_period = self.inter_period_operators_enabled and d_idx >= self.ip_destroy_start

            if is_inter_period:
                # 1. Inter-period destroy
                destroy_op = self.destroy_ops[d_idx]
                n_remove_nodes = self.random.randint(1, max(1, self.params.min_removal))
                new_horizon_candidate, removed = destroy_op(
                    [[[*r] for r in day] for day in current_horizon], n_remove_nodes
                )

                # 2. Inter-period repair (select from IP repair pool)
                ip_repair_weights = self.repair_weights[self.ip_repair_start :]
                r_idx_rel = self.select_operator(ip_repair_weights)
                r_idx = self.ip_repair_start + r_idx_rel
                repair_op = self.repair_ops[r_idx]

                new_horizon_candidate = repair_op(new_horizon_candidate, removed)
            else:
                # Single-day operator: choose a random day and apply parent logic
                t = self.random.randint(0, self.horizon - 1)
                day_routes = [r[:] for r in current_horizon[t]]
                new_day_routes, removed_nodes = self._apply_single_day_ops(day_routes, d_idx)
                new_horizon_candidate = [[[*r] for r in d] for d in current_horizon]
                new_horizon_candidate[t] = new_day_routes

            new_cost = self.calculate_horizon_cost(new_horizon_candidate)
            new_profit = self.calculate_horizon_profit(new_horizon_candidate)

            accept_result = self._check_accept(current_profit, new_profit, best_profit, _it)
            accept = accept_result[0] if isinstance(accept_result, tuple) else accept_result

            if isinstance(accept_result, tuple):
                accept, _ = accept_result

            score = 0.0
            if new_profit > best_profit + 1e-6:
                best_horizon = [[[*r] for r in day] for day in new_horizon_candidate]
                best_profit = new_profit
                best_cost = new_cost
                score = self.params.sigma_1
            elif accept:
                score = self.params.sigma_2 if new_profit > current_profit + 1e-6 else self.params.sigma_3

            if accept:
                current_horizon = new_horizon_candidate
                current_profit = new_profit

            if not is_inter_period:
                r_idx = self.select_operator(self.repair_weights[: self.ip_repair_start])

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

        return best_horizon, best_profit, best_cost

    def _apply_single_day_ops(self, day_routes: List[List[int]], d_idx: int) -> Tuple[List[List[int]], List[int]]:
        """Apply a single-day destroy operator and greedily repair."""
        destroy_op = self.destroy_ops[d_idx]
        n_nodes = sum(len(r) for r in day_routes)
        if n_nodes == 0:
            return day_routes, []

        lower = min(n_nodes, self.params.min_removal)
        upper = min(n_nodes, self.params.max_removal_cap, max(lower, int(n_nodes * self.params.max_removal_pct)))
        upper = max(upper, lower)
        n_rem = self.random.randint(lower, upper)

        new_routes, removed = destroy_op([r[:] for r in day_routes], n_rem)

        # Repair with greedy (single-day)
        common_kw: Dict = dict(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.vrpp,
        )
        if self.profit_aware_operators:
            repaired = greedy_profit_insertion(new_routes, removed, R=self.R, C=self.C, **common_kw)
        else:
            repaired = greedy_insertion(new_routes, removed, **common_kw)

        return repaired, removed

    def _check_accept(
        self, current_profit: float, new_profit: float, best_profit: float, iteration: int
    ) -> Tuple[bool, Any]:
        """Delegate acceptance check to the criterion from the parent class."""
        if self.acceptance_criterion is None:
            return (new_profit > current_profit + 1e-6, None)
        return self.acceptance_criterion.accept(  # type: ignore[return-value]
            current_obj=current_profit,
            candidate_obj=new_profit,
            f_best=best_profit,
            iteration=iteration,
            max_iterations=self.params.max_iterations,
        )
