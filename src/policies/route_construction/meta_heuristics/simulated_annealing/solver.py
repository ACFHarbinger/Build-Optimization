"""
Simulated Annealing (SA) core solver for VRPP.

This implementation executes a discrete-time Markov chain traversal of the
routing solution space. It evaluates candidate neighborhood structures utilizing
the stochastic Metropolis acceptance criterion, mathematically formulated to
allow controlled objective deterioration to escape local optima.

The objective function f(s) evaluates Net Profit. As this is a maximization
problem, the Boltzmann acceptance probability for a deteriorating move (Δf < 0)
is defined exactly as: P = exp(Δf / T).

References:
    Kirkpatrick, S., et al. (1983). "Optimization by simulated annealing". Science.
    Metropolis, N., et al. (1953). "Equation of state calculations by fast computing machines".
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from logic.src.policies.helpers.local_search.local_search_base import LocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes

from .params import SAParams


class SASolver(LocalSearch):
    """
    Simulated Annealing exact solver logic.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        super().__init__(
            dist_matrix=dist_matrix,
            waste=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
        )
        self.mandatory_nodes = mandatory_nodes or []
        self.nodes = list(range(1, len(dist_matrix)))
        self.thermal_log: List[Dict[str, Any]] = []
        self.acceptance_criterion = params.acceptance_criterion

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculate net profit: Revenue - Cost."""
        if not routes:
            return 0.0
        revenue = sum(self.waste.get(node, 0.0) for r in routes for node in r)
        return (revenue * self.R) - (self._cost(routes) * self.C)

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            prev = 0
            for node in route:
                total += self.d[prev, node]
                prev = node
            total += self.d[prev, 0]
        return total

    def _get_route_arcs(self, route: List[int]) -> List[Tuple[int, int]]:
        """Helper to extract arcs from a route."""
        if not route:
            return []
        arcs = [(0, route[0])]
        for i in range(len(route) - 1):
            arcs.append((route[i], route[i + 1]))
        arcs.append((route[-1], 0))
        return arcs

    def _get_affected_arcs(
        self, op: str, current_routes: List[List[int]], u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Fix 7: Extract removed and inserted arcs for an operator."""
        removed = []
        inserted = []

        route_u = current_routes[r_u]
        u_prev = route_u[p_u - 1] if p_u > 0 else 0
        u_next = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
        if op == "2opt":
            # Segment [p_u+1, p_v] reversed. p_u < p_v
            v_next = route_u[p_v + 1] if p_v < len(route_u) - 1 else 0
            removed = [(u, route_u[p_u + 1]), (route_u[p_v], v_next)]
            inserted = [(u, route_u[p_v]), (route_u[p_u + 1], v_next)]
        elif op == "swap":
            route_v = current_routes[r_v]
            v_prev = route_v[p_v - 1] if p_v > 0 else 0
            v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0
            if r_u == r_v and abs(p_u - p_v) == 1:
                # Adjacent nodes
                i, j = (p_u, p_v) if p_u < p_v else (p_v, p_u)
                prev = route_u[i - 1] if i > 0 else 0
                nxt = route_u[j + 1] if j < len(route_u) - 1 else 0
                removed = [(prev, route_u[i]), (route_u[i], route_u[j]), (route_u[j], nxt)]
                inserted = [(prev, route_u[j]), (route_u[j], route_u[i]), (route_u[i], nxt)]
            else:
                removed = [(u_prev, u), (u, u_next), (v_prev, v), (v, v_next)]
                inserted = [(u_prev, v), (v, u_next), (v_prev, u), (u, v_next)]
        elif op == "relocate":
            route_v = current_routes[r_v]
            v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0
            removed = [(u_prev, u), (u, u_next), (v, v_next)]
            inserted = [(u_prev, u_next), (v, u), (u, v_next)]

        return removed, inserted

    def _delta_evaluate(self, removed: List[Tuple[int, int]], inserted: List[Tuple[int, int]]) -> float:
        """Calculate profit delta from arc changes."""
        dist_delta = sum(self.d[i, j] for i, j in inserted) - sum(self.d[i, j] for i, j in removed)
        return -dist_delta * self.C

    def _perturb(self, current_routes: List[List[int]], T: float, T_0: float) -> Tuple[str, List[List[int]], float]:
        """
        Generate neighbor, returning (op, solution, delta_profit).
        """
        roll = self.random.random()
        if roll < 0.1:
            op = "ruin_recreate"
            n_remove = max(1, int((T / T_0) * (len(self.nodes) // 5)))
            n_remove = min(n_remove, max(1, len(self.nodes) // 2))

            copied = [list(r) for r in current_routes]
            partial, removed = random_removal(copied, n_remove, rng=self.random)
            if self.params.profit_aware_operators:
                new_sol = greedy_profit_insertion(
                    partial,
                    removed,
                    self.d,
                    self.waste,
                    self.Q,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                new_sol = greedy_insertion(
                    partial,
                    removed,
                    self.d,
                    self.waste,
                    self.Q,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            return op, new_sol, self._evaluate(new_sol) - self._evaluate(current_routes)

        active_nodes = [n for r in current_routes for n in r]
        if not active_nodes:
            new_sol = build_greedy_routes(self.d, self.waste, self.Q, self.R, self.C, self.mandatory_nodes, self.random)
            return "reinit", new_sol, self._evaluate(new_sol)

        u = self.random.choice(active_nodes)
        v = self.random.choice(self.neighbors[u]) if u in self.neighbors else self.random.choice(active_nodes)

        r_u, p_u = self.node_map[u]
        v_loc = self.node_map.get(v)
        copied = [list(r) for r in current_routes]

        if v_loc:
            r_v, p_v = v_loc
            if r_u == r_v:
                if self.random.random() < 0.5:
                    op = "2opt"
                    i, j = (p_u, p_v) if p_u < p_v else (p_v, p_u)
                    if j - i < 2:
                        return "noop", current_routes, 0.0
                    rem, ins = self._get_affected_arcs(op, current_routes, u, v, r_u, i, r_u, j)
                    copied[r_u][i + 1 : j + 1] = copied[r_u][i + 1 : j + 1][::-1]
                    return op, copied, self._delta_evaluate(rem, ins)
                else:
                    op = "swap"
                    rem, ins = self._get_affected_arcs(op, current_routes, u, v, r_u, p_u, r_v, p_v)
                    copied[r_u][p_u], copied[r_v][p_v] = copied[r_v][p_v], copied[r_u][p_u]
                    return op, copied, self._delta_evaluate(rem, ins)
            else:
                op = "relocate"
                if sum(self.waste.get(n, 0) for n in copied[r_v]) + self.waste.get(u, 0) > self.Q:
                    return "noop", current_routes, 0.0
                rem, ins = self._get_affected_arcs(op, current_routes, u, v, r_u, p_u, r_v, p_v)
                node = copied[r_u].pop(p_u)
                copied[r_v].insert(p_v + 1, node)
                return op, copied, self._delta_evaluate(rem, ins)
        else:
            # v is unvisited
            op = "swap_unvisited"
            if self._calc_load_fresh(copied[r_u]) - self.waste.get(u, 0) + self.waste.get(v, 0) > self.Q:
                return "noop", current_routes, 0.0
            rev_gain = (self.waste.get(v, 0) - self.waste.get(u, 0)) * self.R
            u_prev = copied[r_u][p_u - 1] if p_u > 0 else 0
            u_next = copied[r_u][p_u + 1] if p_u < len(copied[r_u]) - 1 else 0
            dist_change = (self.d[u_prev, v] + self.d[v, u_next]) - (self.d[u_prev, u] + self.d[u, u_next])
            copied[r_u][p_u] = v
            return op, copied, rev_gain - (dist_change * self.C)

    def _calibrate_initial_temperature(self, start_routes: List[List[int]]) -> float:
        """Solve for T_0 such that initial acceptance of worsening moves is approx 80%."""
        worsening_deltas = []
        for _ in range(self.params.calibration_samples):
            _, _, delta_f = self._perturb(
                start_routes, self.params.initial_temperature, self.params.initial_temperature
            )
            if delta_f < -1e-4:
                worsening_deltas.append(abs(delta_f))

        if not worsening_deltas:
            return self.params.initial_temperature

        avg_w = sum(worsening_deltas) / len(worsening_deltas)
        return -avg_w / math.log(self.params.target_initial_acceptance)

    def optimize(self, solution: List[List[int]]) -> List[List[int]]:
        """Implementation for LocalSearch ABC."""
        best_routes, _, __ = self.solve(initial_solution=solution)
        return best_routes

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:  # noqa: C901
        """
        Kirkpatrick (1983) Simulated Annealing.
        Adaptive Markov chains, specific heat tracking, and multi-start.
        """
        start_time = time.time()
        global_best_routes = []
        global_best_profit = -float("inf")

        for r_idx in range(self.params.n_restarts):
            if time.time() - start_time >= self.params.time_limit:
                break

            # Initial state
            if initial_solution and r_idx == 0:
                current_routes = [list(r) for r in initial_solution]
            else:
                current_routes = build_greedy_routes(
                    self.d, self.waste, self.Q, self.R, self.C, self.mandatory_nodes, self.random
                )

            self.routes = current_routes
            self._update_map(set(range(len(self.routes))))
            current_profit = self._evaluate(current_routes)

            best_routes = [list(r) for r in current_routes]
            best_profit = current_profit

            assert self.acceptance_criterion is not None, "Acceptance criterion must be provided."
            if self.params.auto_calibrate_temperature:
                T_calib = self._calibrate_initial_temperature(current_routes)
                # If using Boltzmann, update its temperature
                if hasattr(self.acceptance_criterion, "T"):
                    self.acceptance_criterion.T = T_calib

            self.acceptance_criterion.setup(current_profit)
            frozen_streak = 0

            # Use a generic termination condition; some criteria may override this
            # For now, we continue using T-based termination for compatibility if Boltzmann
            while frozen_streak < self.params.frozen_streak_limit:
                # Basic check for termination from criterion if applicable
                state = self.acceptance_criterion.get_state()
                T_curr = state.get("temperature", 1.0)  # Fallback to avoid infinite loops if no temp
                if hasattr(self.acceptance_criterion, "T") and T_curr < self.params.min_temperature:
                    break
                if time.time() - start_time >= self.params.time_limit:
                    break

                n_accepted = 0
                n_attempts = 0
                target_accept = self.params.target_acceptances_per_node * len(self.nodes)
                max_attempts = self.params.max_attempts_multiplier * len(self.nodes)

                energies = []

                while n_accepted < target_accept and n_attempts < max_attempts:
                    if time.time() - start_time >= self.params.time_limit:
                        break

                    current_state = self.acceptance_criterion.get_state()
                    T_curr = current_state.get("temperature", 1.0)
                    T_0_curr = current_state.get("initial_temperature", 1.0)

                    op, candidate, delta_f = self._perturb(current_routes, T_curr, T_0_curr)
                    n_attempts += 1

                    if self.acceptance_criterion.accept(
                        current_obj=current_profit,
                        candidate_obj=current_profit + delta_f,
                        f_best=best_profit,
                        iteration=n_attempts,
                        max_iterations=max_attempts,
                    ):
                        current_routes = candidate
                        current_profit += delta_f
                        n_accepted += 1
                        accepted_this_move = True
                    else:
                        accepted_this_move = False

                    self.acceptance_criterion.step(
                        current_obj=current_profit,
                        candidate_obj=current_profit + delta_f,
                        accepted=accepted_this_move,
                        f_best=best_profit,
                        iteration=n_attempts,
                        max_iterations=max_attempts,
                    )

                    if accepted_this_move:
                        # Sync LocalSearch state
                        self.routes = current_routes
                        self._update_map(set(range(len(self.routes))))

                        if current_profit > best_profit:
                            best_profit = current_profit
                            best_routes = [list(r) for r in current_routes]

                    energies.append(current_profit)

                # Cooling and diagnostics
                if n_accepted == 0:
                    frozen_streak += 1
                else:
                    frozen_streak = 0

                if len(energies) > 1:
                    var = np.var(energies)
                    assert self.acceptance_criterion is not None
                    current_state = self.acceptance_criterion.get_state()
                    T_diag = current_state.get("temperature", 0.0)
                    c_spec = var / (T_diag**2) if T_diag > 0 else 0.0
                    self.thermal_log.append(
                        {
                            "T": T_diag,
                            "mean_f": np.mean(energies),
                            "C": c_spec,
                            "acc": n_accepted,
                            "state": current_state,
                        }
                    )

                # Note: self.acceptance_criterion.step() handles internal cooling or state updates

            if best_profit > global_best_profit:
                global_best_profit = best_profit
                global_best_routes = best_routes

        return global_best_routes, global_best_profit, self._cost(global_best_routes)
