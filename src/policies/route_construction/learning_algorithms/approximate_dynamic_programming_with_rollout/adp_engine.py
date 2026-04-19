"""
ADP Rollout Engine for Stochastic Multi-Period IRP.

Implements Powell's Approximate Dynamic Programming framework adapted to
the Stochastic Inventory Routing Problem (SIRP).  At each day ``t``, the
engine enumerates candidate node sets, evaluates the combined routing cost
plus a truncated rollout estimate of future costs, and selects the best set.

Algorithm (per day t)
---------------------
1. Generate candidate sets S ⊆ V using ``candidate_strategy``.
2. For each S:
   a. Solve a fast TSP/VRP on S → ``routing_cost_t(S)``
   b. Simulate demand for t+1 … t+H (N monte-carlo scenarios)
   c. Run the greedy insertion baseline on each simulated state
   d. Sum: ``ΔObj(S) = routing_cost_t(S) + V̂(S_{t+1}) − revenue(S)``
3. Select ``S* = argmin ΔObj``
4. Execute S*, update inventory state.

References
----------
Powell, W. B. (2011). "Approximate Dynamic Programming: Solving the Curses
of Dimensionality." Wiley-Interscience.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators import greedy_insertion

from .params import ADPRolloutParams


class ADPRolloutEngine:
    """Approximate Dynamic Programming engine for SIRP.

    Uses a greedy rollout baseline to estimate the expected future value
    V(S_{t+1}) under candidate visit decisions, then selects the visit set
    that minimises total expected cost across the look-ahead horizon H.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ADPRolloutParams,
        mandatory_nodes: Optional[List[int]] = None,
    ) -> None:
        """Initialise the ADP rollout engine.

        Args:
            dist_matrix: Square distance/cost matrix (0-indexed, depot = 0).
            wastes: Mapping {node_id: current_fill_level_%} for all customers.
            capacity: Vehicle capacity.
            R: Revenue per unit waste collected.
            C: Cost per unit distance.
            params: ADP rollout configuration.
            mandatory_nodes: Nodes that must be served regardless of fill level.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes: List[int] = mandatory_nodes or []
        self.rng = random.Random(params.seed)
        self.np_rng = np.random.default_rng(params.seed)
        self.n_nodes = len(dist_matrix) - 1

    # ------------------------------------------------------------------
    # Candidate set generation
    # ------------------------------------------------------------------

    def _generate_candidates(self, wastes: Dict[int, float]) -> List[List[int]]:
        """Generate candidate node sets using the configured strategy.

        Strategies:
            ``"threshold"`` — all nodes with fill ≥ ``fill_threshold``.
            ``"top_k"``     — the K highest-fill nodes.
            ``"beam"``      — ``max_candidate_sets`` random subsets of
                              above-threshold nodes.

        Args:
            wastes: Current fill levels for all nodes.

        Returns:
            List of candidate node sets (each a sorted list of node IDs).
        """
        strategy = self.params.candidate_strategy
        threshold = self.params.fill_threshold

        # Always include mandatory nodes
        mandatory_set: Set[int] = set(self.mandatory_nodes)
        eligible = sorted(
            [n for n, fill in wastes.items() if fill >= threshold or n in mandatory_set],
            key=lambda n: -wastes.get(n, 0.0),
        )

        if not eligible:
            eligible = sorted(mandatory_set)

        if strategy == "threshold":
            return [eligible] if eligible else []

        elif strategy == "top_k":
            k = min(self.params.top_k, len(eligible))
            return [eligible[:k]] if eligible else []

        elif strategy == "beam":
            if not eligible:
                return []
            # Always include one candidate with all eligible nodes
            candidates: List[List[int]] = [eligible[:]]
            max_sets = self.params.max_candidate_sets - 1
            for _ in range(max_sets):
                # Sample a random subset of eligible nodes
                size = self.rng.randint(max(1, len(mandatory_set)), len(eligible))
                subset = sorted(self.rng.sample(eligible, size))
                # Ensure mandatory nodes are always included
                subset = sorted(set(subset) | mandatory_set)
                if subset not in candidates:
                    candidates.append(subset)
            return candidates

        else:
            raise ValueError(f"Unknown candidate_strategy: '{strategy}'")

    # ------------------------------------------------------------------
    # Routing cost evaluation
    # ------------------------------------------------------------------

    def _route_cost(self, candidate_nodes: List[int], wastes: Dict[int, float]) -> float:
        """Estimate routing cost using greedy insertion.

        Args:
            candidate_nodes: Nodes to visit.
            wastes: Current fill levels (used for capacity feasibility).

        Returns:
            Total routing cost (C * total_distance).
        """
        if not candidate_nodes:
            return 0.0

        # Greedy insertion produces routes; cost = sum of distances × C
        routes = greedy_insertion(
            routes=[],
            removed_nodes=list(candidate_nodes),
            dist_matrix=self.dist_matrix,
            wastes=wastes,
            capacity=self.capacity,
            expand_pool=True,
        )
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            total_dist += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total_dist += self.dist_matrix[route[i]][route[i + 1]]
            total_dist += self.dist_matrix[route[-1]][0]

        return total_dist * self.C

    def _revenue(self, candidate_nodes: List[int], wastes: Dict[int, float]) -> float:
        """Compute total revenue from collecting a candidate set.

        Args:
            candidate_nodes: Nodes to collect.
            wastes: Current fill levels (% of capacity).

        Returns:
            Total revenue = Σ_{i ∈ S} wastes[i] * R.
        """
        return sum(wastes.get(n, 0.0) * self.R for n in candidate_nodes)

    # ------------------------------------------------------------------
    # Rollout simulation
    # ------------------------------------------------------------------

    def _simulate_forward(
        self,
        wastes: Dict[int, float],
        visited_today: Set[int],
        scenario_tree: Optional[Any],
        t_current: int,
        t_total: int,
    ) -> float:
        """Estimate future cost E[V(S_{t+1})] via greedy rollout.

        Samples ``n_scenarios`` demand realisations from ``scenario_tree``
        (or uses a mean heuristic if no tree is available), then applies
        a fast greedy baseline policy on each scenario.

        Args:
            wastes: Current fill levels after today's collection.
            visited_today: Nodes visited on day ``t_current``.
            scenario_tree: Optional ScenarioTree for demand sampling.
            t_current: Current day index.
            t_total: Total simulation horizon T.

        Returns:
            Weighted average future cost.
        """
        if t_current >= t_total - 1 or self.params.look_ahead_days == 0:
            return 0.0

        H = min(self.params.look_ahead_days, t_total - t_current - 1)

        # Simulate post-collection inventory
        post_wastes = {n: (0.0 if n in visited_today else wastes.get(n, 0.0)) for n in wastes}

        total_future_cost = 0.0
        n_valid = 0

        for _ in range(self.params.n_scenarios):
            scenario_wastes = dict(post_wastes)
            future_cost = 0.0

            for step in range(H):
                # Demand evolution
                if scenario_tree is not None and hasattr(scenario_tree, "get_scenarios_at_day"):
                    day_scenarios = scenario_tree.get_scenarios_at_day(t_current + step + 1)
                    if day_scenarios:
                        sc = self.rng.choice(day_scenarios)
                        for n in scenario_wastes:
                            if hasattr(sc, "wastes") and n - 1 < len(sc.wastes):
                                delta = float(sc.wastes[n - 1])
                            else:
                                delta = self.np_rng.uniform(0.0, 20.0)
                            scenario_wastes[n] = min(scenario_wastes[n] + delta, 200.0)
                    else:
                        # Random increment
                        for n in scenario_wastes:
                            scenario_wastes[n] = min(scenario_wastes[n] + self.np_rng.uniform(0.0, 15.0), 200.0)
                else:
                    for n in scenario_wastes:
                        scenario_wastes[n] = min(scenario_wastes[n] + self.np_rng.uniform(0.0, 15.0), 200.0)

                # Overflow penalty before next collection
                for _n, fill in scenario_wastes.items():
                    overflow = max(0.0, fill - 100.0)
                    future_cost += overflow * self.params.stockout_penalty

                # Greedy collection
                eligible_future = [n for n, fill in scenario_wastes.items() if fill >= self.params.fill_threshold]
                mandatory_future = [n for n in self.mandatory_nodes if n in scenario_wastes]
                to_visit = sorted(set(eligible_future) | set(mandatory_future))

                if to_visit:
                    future_cost += self._route_cost(to_visit, scenario_wastes)
                    for n in to_visit:
                        scenario_wastes[n] = 0.0

            total_future_cost += future_cost
            n_valid += 1

        return total_future_cost / n_valid if n_valid > 0 else 0.0

    # ------------------------------------------------------------------
    # Single-day optimisation
    # ------------------------------------------------------------------

    def solve_day(
        self,
        wastes: Dict[int, float],
        scenario_tree: Optional[Any],
        t_current: int,
        t_total: int,
    ) -> Tuple[List[List[int]], float, float, Dict[str, Any]]:
        """Select the best candidate set and solve routing for one day.

        Args:
            wastes: Current fill levels for all nodes.
            scenario_tree: ScenarioTree for forward value estimation.
            t_current: Current day index.
            t_total: Total planning horizon T.

        Returns:
            Tuple of:
                - routes: List of routes for the selected candidate set.
                - routing_cost: Day-t routing cost.
                - profit: Revenue − routing_cost.
                - metadata: Dict with selected candidate set and ADP value.
        """
        candidate_sets = self._generate_candidates(wastes)

        if not candidate_sets:
            return [], 0.0, 0.0, {"candidate_set": [], "approx_value": 0.0}

        best_obj = float("inf")
        best_candidates: List[int] = []
        best_routing_cost = 0.0
        best_future_value = 0.0

        for candidates in candidate_sets:
            if not candidates:
                continue
            routing_cost = self._route_cost(candidates, wastes)
            revenue = self._revenue(candidates, wastes)
            visited_set: Set[int] = set(candidates)
            future_value = self._simulate_forward(wastes, visited_set, scenario_tree, t_current, t_total)
            # ADP objective: minimise (routing_cost + future_value − revenue)
            obj = routing_cost + future_value - revenue

            if obj < best_obj:
                best_obj = obj
                best_candidates = candidates
                best_routing_cost = routing_cost
                best_future_value = future_value

        # Build actual routes for the chosen candidate set
        if best_candidates:
            routes = greedy_insertion(
                routes=[],
                removed_nodes=list(best_candidates),
                dist_matrix=self.dist_matrix,
                wastes=wastes,
                capacity=self.capacity,
                expand_pool=True,
            )
        else:
            routes = []

        best_revenue = self._revenue(best_candidates, wastes)
        profit = best_revenue - best_routing_cost

        metadata: Dict[str, Any] = {
            "candidate_set": best_candidates,
            "approx_value": best_future_value,
            "n_candidate_sets_evaluated": len(candidate_sets),
        }

        return routes, best_routing_cost, profit, metadata

    # ------------------------------------------------------------------
    # Full multi-period solve
    # ------------------------------------------------------------------

    def solve(
        self,
        scenario_tree: Optional[Any],
        horizon: int,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """Solve the full T-day horizon via rolling ADP.

        Args:
            scenario_tree: ScenarioTree for demand simulation.
            horizon: Planning horizon T.

        Returns:
            Tuple of:
                - full_plan[day][route][node]: T-day routing plan.
                - total_expected_profit: Σ_t profit_t.
                - metadata: Summary statistics.
        """
        start_time = time.process_time()
        full_plan: List[List[List[int]]] = []
        total_profit = 0.0
        rolling_wastes = dict(self.wastes)

        for t in range(horizon):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                # Fill remaining days with empty routes
                full_plan.extend([[] for _ in range(horizon - t)])
                break

            routes, routing_cost, profit, day_meta = self.solve_day(
                wastes=rolling_wastes,
                scenario_tree=scenario_tree,
                t_current=t,
                t_total=horizon,
            )
            full_plan.append(routes)
            total_profit += profit

            # Update rolling inventory: collected nodes reset to 0
            visited: Set[int] = {n for r in routes for n in r}
            for n in rolling_wastes:
                if n in visited:
                    rolling_wastes[n] = 0.0

            if self.params.verbose:
                print(f"[ADP] Day {t}: profit={profit:.2f}, nodes={day_meta['candidate_set']}")

        metadata: Dict[str, Any] = {
            "total_profit": total_profit,
            "wall_time": time.process_time() - start_time,
        }
        return full_plan, total_profit, metadata
