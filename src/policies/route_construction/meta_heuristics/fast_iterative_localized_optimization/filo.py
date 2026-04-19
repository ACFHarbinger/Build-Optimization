"""
Fast Iterative Localized Optimization (FILO) policy module.

This module provides the main entry point for the FILO metaheuristic,
incorporating Ruin & Recreate shaking and Local Search via Simulated Annealing.

References:
    Accorsi, L., & Vigo, D. "A fast and scalable heuristic for the solution
    of large-scale capacitated vehicle routing problems", 2021.
"""

import copy
import math
import random
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_filo import (
    FILOLocalSearch,
)
from logic.src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.params import (
    FILOParams,
)
from logic.src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.ruin_recreate import (
    RuinAndRecreate,
)


class FILOSolver:
    """
    Implementation of Fast Iterative Localized Optimization (FILO) for CVRP/VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: FILOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initialize the FILO solver."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.all_customers = [n for n in self.waste.keys() if n != 0]
        self.random = random.Random(self.params.seed)
        self.rng = np.random.default_rng(self.params.seed)

        self.ruin_recreate = RuinAndRecreate(
            dist_matrix=self.d,
            wastes=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            rng=self.rng,
            profit_aware_operators=self.params.profit_aware_operators,
            vrpp=self.params.vrpp,
        )

        self.local_search = FILOLocalSearch(
            dist_matrix=self.d,
            waste=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            params=self.params,
        )

        self.node_gamma = [self.params.gamma_base] * (self.n_nodes + 1)
        self.non_improving_iter = [0] * (self.n_nodes + 1)
        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))
        self.node_omega = [omega_base] * (self.n_nodes + 1)
        self.svc_cache: OrderedDict[int, None] = OrderedDict()
        self.svc_history: List[int] = []

    def _evaluate_routes(self, routes: List[List[int]]) -> Tuple[float, float]:
        """Evaluate VRPP cost and profit."""
        total_cost, total_revenue = 0.0, 0.0
        for route in routes:
            if not route:
                continue
            prev = 0
            for node in route:
                total_cost += self.d[prev, node] * self.C
                total_revenue += self.waste.get(node, 0.0) * self.R
                prev = node
            total_cost += self.d[prev, 0] * self.C
        return total_cost, total_revenue - total_cost

    def _update_gamma(self, is_new_best: bool, improved: bool, ruined_and_recreated: List[int]) -> None:
        """Update localized gamma parameters (expansion strategy)."""
        if is_new_best:
            for i in ruined_and_recreated:
                self.node_gamma[i], self.non_improving_iter[i] = self.params.gamma_base, 0
            return

        avg_svc = sum(self.svc_history) / len(self.svc_history) if self.svc_history else self.params.svc_size
        dynamic_threshold = max(1, int((self.params.delta_gamma * self.params.max_iterations * avg_svc) / self.n_nodes))

        for i in ruined_and_recreated:
            if not improved:
                self.non_improving_iter[i] += 1
                if self.non_improving_iter[i] >= dynamic_threshold:
                    self.node_gamma[i] = min(1.0, self.node_gamma[i] * self.params.gamma_lambda)
                    self.non_improving_iter[i] = 0
            else:
                self.node_gamma[i], self.non_improving_iter[i] = self.params.gamma_base, 0

    def _update_omega(
        self, current_cost: float, routes: List[List[int]], delta_profit: float, ruined_and_recreated: List[int]
    ) -> None:
        """Update localized omega meta-strategy strictly targeting ruined nodes."""
        n_visited = sum(len(r) for r in routes)
        n_routes = len(routes)
        if n_visited == 0:
            return
        c_bar = current_cost / (n_visited + 2 * n_routes)
        omega_lb, omega_ub = c_bar * self.params.shaking_lb_intensity, c_bar * self.params.shaking_ub_intensity
        delta_cost = -delta_profit
        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))
        for i in ruined_and_recreated:
            if 0 <= delta_cost < omega_lb:
                self.node_omega[i] = min(50, self.node_omega[i] + 1)
            elif delta_cost > omega_ub:
                if self.node_omega[i] > omega_base:
                    self.node_omega[i] -= 1
            else:
                if self.rng.random() < 0.5:
                    self.node_omega[i] = min(50, self.node_omega[i] + 1)
                elif self.node_omega[i] > omega_base:
                    self.node_omega[i] -= 1

    def _update_svc(self, nodes: List[int]) -> None:
        """Update Selective Vertex Caching (LRU)."""
        for node in nodes:
            if node in self.svc_cache:
                self.svc_cache.move_to_end(node)
            else:
                self.svc_cache[node] = None
                if len(self.svc_cache) > self.params.svc_size:
                    self.svc_cache.popitem(last=False)
        self.svc_history.append(len(self.svc_cache))

    def _clarke_wright_initialization(self) -> List[List[int]]:
        """Linearized Clarke and Wright savings algorithm."""
        savings, n_cw = [], self.params.n_cw
        for i in self.all_customers:
            count = 0
            for j in self.ruin_recreate.neighbors[i]:
                if count >= n_cw:
                    break
                if j > i:
                    savings.append((self.d[i, 0] + self.d[0, j] - self.d[i, j], i, j))
                    count += 1
        savings.sort(key=lambda x: x[0], reverse=True)
        node_to_route = {i: [i] for i in self.all_customers}
        routes, route_load = (
            list(node_to_route.values()),
            {id(r): self.waste.get(r[0], 0.0) for r in node_to_route.values()},
        )
        for _, i, j in savings:
            ri, rj = node_to_route[i], node_to_route[j]
            if ri is rj or route_load[id(ri)] + route_load[id(rj)] > self.Q:
                continue
            if ri[-1] == i and rj[0] == j:
                ri.extend(rj)
                for node in rj:
                    node_to_route[node] = ri
                route_load[id(ri)] += route_load[id(rj)]
                routes.remove(rj)
            elif rj[-1] == j and ri[0] == i:
                rj.extend(ri)
                for node in ri:
                    node_to_route[node] = rj
                route_load[id(rj)] += route_load[id(ri)]
                routes.remove(ri)
        return [r for r in routes if r]

    def _find_best_insertion(self, working_routes: List[List[int]], customer: int) -> Tuple[int, int]:
        """Find the best route index and position for a customer."""
        best_cost_inc, best_ri, best_pos = float("inf"), -1, -1
        w = self.waste.get(customer, 0.0)
        for ri, r in enumerate(working_routes):
            if sum(self.waste.get(n, 0.0) for n in r) + w > self.Q:
                continue
            for pos in range(len(r) + 1):
                prev, nxt = r[pos - 1] if pos > 0 else 0, r[pos] if pos < len(r) else 0
                cost_inc = self.d[prev, customer] + self.d[customer, nxt] - self.d[prev, nxt]
                if cost_inc < best_cost_inc:
                    best_cost_inc, best_ri, best_pos = cost_inc, ri, pos
        return best_ri, best_pos

    def _route_minimization(self, routes: List[List[int]]) -> List[List[int]]:
        """Algorithm 4: Episodic Route Minimization."""
        if not routes or len(routes) < 2:
            return routes

        working_routes = [r[:] for r in routes if r]
        delta_rm = len(working_routes)

        P = 1.0
        P_f = 0.01
        z = (P_f / P) ** (1.0 / delta_rm) if delta_rm > 0 else 1.0

        for _ in range(delta_rm):
            if len(working_routes) < 2:
                break

            # 1. Select seed route
            r1_idx = self.rng.choice(len(working_routes))
            seed_route = working_routes[r1_idx]

            # 2. Find proximity neighbor route
            best_dist = float("inf")
            r2_idx = -1

            if seed_route:
                seed_node = self.rng.choice(seed_route)
                for r_idx, r in enumerate(working_routes):
                    if r_idx == r1_idx or not r:
                        continue
                    dist = min(self.d[seed_node, n] for n in r)
                    if dist < best_dist:
                        best_dist = dist
                        r2_idx = r_idx

            if r2_idx == -1:
                r2_idx = (r1_idx + 1) % len(working_routes)

            idx1, idx2 = min(r1_idx, r2_idx), max(r1_idx, r2_idx)
            route2 = working_routes.pop(idx2)
            route1 = working_routes.pop(idx1)

            unrouted = route1 + route2
            self.rng.shuffle(unrouted)
            deferred = []

            for customer in unrouted:
                best_ri, best_pos = self._find_best_insertion(working_routes, customer)
                if best_ri != -1:
                    working_routes[best_ri].insert(best_pos, customer)
                else:
                    if self.rng.random() > P:
                        deferred.append(customer)
                    else:
                        working_routes.append([customer])

            for customer in deferred:
                best_ri, best_pos = self._find_best_insertion(working_routes, customer)
                if best_ri != -1:
                    working_routes[best_ri].insert(best_pos, customer)
                else:
                    working_routes.append([customer])

            P *= z

            # Restricted LS tightening per episode
            affected = set(unrouted)
            working_routes = self.local_search.optimize(
                working_routes, active_nodes=affected, node_gamma=self.node_gamma
            )
            working_routes = [r for r in working_routes if r]

        return [r for r in working_routes if r]

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Execute the FILO heuristic."""
        start_time = time.process_time()
        current_routes = self._clarke_wright_initialization()
        current_routes = self._route_minimization(current_routes)
        current_cost, current_profit = self._evaluate_routes(current_routes)
        best_routes, best_profit, best_cost = copy.deepcopy(current_routes), current_profit, current_cost
        if current_cost > 0:
            self.sa_start_temp, self.sa_final_temp = (
                current_cost / self.params.initial_temperature_factor,
                current_cost / self.params.final_temperature_factor,
            )
        else:
            self.sa_start_temp, self.sa_final_temp = 100.0, 1.0
        temperature = self.sa_start_temp
        for _iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break
            visited = [n for r in current_routes for n in r]
            if not visited:
                break
            gammas = np.array([self.node_gamma[n] for n in visited], dtype=np.float64)
            center_node = self.rng.choice(visited, p=gammas / gammas.sum())
            l_it = self.node_omega[center_node]
            # Pass only the single center node. The walk length is defined by l_it.
            new_routes, _, footprint_S = self.ruin_recreate.apply(
                routes=current_routes,
                seed=center_node,
                all_customers=self.all_customers,
                mandatory_nodes=[],
                omega_intensity=float(l_it),
            )
            self._update_svc(footprint_S)
            ls_routes = self.local_search.optimize(
                new_routes, active_nodes=set(self.svc_cache.keys()), node_gamma=self.node_gamma
            )
            ls_cost, ls_profit = self._evaluate_routes(ls_routes)
            delta_profit = ls_profit - current_profit
            accept = (delta_profit > 1e-6) or (
                temperature > 0 and self.rng.random() < math.exp(delta_profit / temperature)
            )
            is_new_best = False
            if ls_profit > best_profit + 1e-6:
                best_routes, best_profit, best_cost, is_new_best = copy.deepcopy(ls_routes), ls_profit, ls_cost, True

            # Shaking strategy strictly targets the returned footprint S
            self._update_gamma(is_new_best, ls_profit > current_profit, footprint_S)
            self._update_omega(current_cost, current_routes, delta_profit, footprint_S)
            if accept:
                current_routes, current_profit, current_cost = ls_routes, ls_profit, ls_cost
            if temperature > self.sa_final_temp:
                temperature *= (self.sa_final_temp / self.sa_start_temp) ** (1.0 / self.params.max_iterations)
        return best_routes, best_profit, best_cost
