"""
Ruin and Recreate operator for Fast Iterative Localized Optimization (FILO).

Upgraded to be strictly Profit-Aware for the VRPP and topologically rigorous.
"""

import copy
from typing import Dict, List, Tuple

import numpy as np


class RuinAndRecreate:
    """
    Operator that applies localized Ruin & Recreate (shaking) to a solution.
    Modified to accurately reflect Accorsi & Vigo (2021) Random Walk.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        rng: np.random.Generator,
        profit_aware_operators: bool = False,
        vrpp: bool = True,
    ):
        """Initialize RuinAndRecreate operator."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.rng = rng
        self.profit_aware_operators = profit_aware_operators
        self.vrpp = vrpp

        # Pre-compute neighbors list
        self.neighbors: List[List[int]] = []
        for i in range(len(self.d)):
            sorted_neighbors = np.argsort(self.d[i])
            filtered = [int(n) for n in sorted_neighbors if n != 0 and n != i]
            self.neighbors.append(filtered)

    def apply(  # noqa: C901
        self,
        routes: List[List[int]],
        seed: int,
        all_customers: List[int],
        mandatory_nodes: List[int],
        omega_intensity: float = 1.0,
    ) -> Tuple[List[List[int]], int, List[int]]:
        """
        Apply Ruin and Recreate to the current routing solution.
        Random Walk length is strictly omega_intensity.
        """
        working_routes = copy.deepcopy(routes)
        current_loads = [sum(self.waste.get(node, 0.0) for node in route) for route in working_routes]
        mandatory_set = set(mandatory_nodes)

        # --- 1. RUIN PHASE (Random Walk) ---
        removed_customers: List[int] = []
        footprint_S = set()

        removed_customers.append(seed)

        curr_node, curr_r_idx, curr_p_idx = seed, -1, -1
        for r_idx, route in enumerate(working_routes):
            if seed in route:
                curr_r_idx, curr_p_idx = r_idx, route.index(seed)
                break

        if curr_r_idx != -1:
            walk_limit = int(omega_intensity)
            while len(removed_customers) < walk_limit:
                if self.rng.random() > 0.5:  # 1 - alpha
                    # Intra-route
                    move = self.rng.choice([-1, 1])
                    new_p = curr_p_idx + move
                    if 0 <= new_p < len(working_routes[curr_r_idx]):
                        curr_p_idx, curr_node = new_p, working_routes[curr_r_idx][new_p]
                        if curr_node not in removed_customers:
                            removed_customers.append(curr_node)
                else:
                    # Inter-route jump
                    found_jump = False
                    for neighbor in self.neighbors[curr_node]:
                        target_r_idx = -1
                        for r_idx, route in enumerate(working_routes):
                            if neighbor in route:
                                target_r_idx = r_idx
                                break
                        if target_r_idx != -1 and target_r_idx != curr_r_idx:
                            curr_r_idx, curr_node = target_r_idx, neighbor
                            curr_p_idx = working_routes[target_r_idx].index(neighbor)
                            if curr_node not in removed_customers:
                                removed_customers.append(curr_node)
                            found_jump = True
                            break
                    if not found_jump:
                        if not removed_customers:
                            break
                        curr_node = int(self.rng.choice(removed_customers))
                        for r_idx, route in enumerate(working_routes):
                            if curr_node in route:
                                curr_r_idx, curr_p_idx = r_idx, route.index(curr_node)
                                break

        # Removal execution & Footprint of broken edges
        for r_idx, route in enumerate(working_routes):
            for i, n in enumerate(route):
                if n in removed_customers:
                    footprint_S.add(n)
                    if i > 0:
                        footprint_S.add(route[i - 1])
                    if i < len(route) - 1:
                        footprint_S.add(route[i + 1])
            working_routes[r_idx] = [n for n in route if n not in removed_customers]
            current_loads[r_idx] = sum(self.waste.get(n, 0.0) for n in working_routes[r_idx])

        # --- 2. RECREATE PHASE ---
        visited = {n for r in working_routes for n in r}
        if self.vrpp:
            reinsertion_pool_set = set(removed_customers)
            for node in removed_customers:
                for neighbor in self.neighbors[node][:15]:
                    if neighbor not in visited:
                        reinsertion_pool_set.add(neighbor)
            reinsertion_pool = list(reinsertion_pool_set)
        else:
            reinsertion_pool = removed_customers

        for customer in reinsertion_pool:
            # Safeguard: Ensure node isn't already re-inserted by a previous step
            if any(customer in r for r in working_routes):
                continue

            best_profit, best_r_idx, best_p_idx = -float("inf"), -2, -1
            w = self.waste.get(customer, 0.0)
            rev, is_mandatory = w * self.R, customer in mandatory_set
            for r_idx, route in enumerate(working_routes):
                if current_loads[r_idx] + w > self.Q:
                    continue
                if not route:
                    continue
                # Start
                d_start = self.d[0, customer] + self.d[customer, route[0]] - self.d[0, route[0]]
                p_start = rev - d_start * self.C
                if p_start > best_profit:
                    best_profit, best_r_idx, best_p_idx = p_start, r_idx, 0
                # Mid
                for p_idx in range(1, len(route)):
                    prev, nxt = route[p_idx - 1], route[p_idx]
                    d_mid = self.d[prev, customer] + self.d[customer, nxt] - self.d[prev, nxt]
                    p_mid = rev - d_mid * self.C
                    if p_mid > best_profit:
                        best_profit, best_r_idx, best_p_idx = p_mid, r_idx, p_idx
                # End
                d_end = self.d[route[-1], customer] + self.d[customer, 0] - self.d[route[-1], 0]
                p_end = rev - d_end * self.C
                if p_end > best_profit:
                    best_profit, best_r_idx, best_p_idx = p_end, r_idx, len(route)
            # New Route
            p_new = rev - (self.d[0, customer] + self.d[customer, 0]) * self.C
            if p_new > best_profit:
                best_profit, best_r_idx = p_new, -1

            if best_r_idx != -2:
                if self.profit_aware_operators and not is_mandatory and best_profit < -1e-4:
                    continue
                if best_r_idx == -1:
                    working_routes.append([customer])
                    current_loads.append(w)
                    footprint_S.add(customer)
                else:
                    footprint_S.add(customer)
                    if best_p_idx > 0:
                        footprint_S.add(working_routes[best_r_idx][best_p_idx - 1])
                    if best_p_idx < len(working_routes[best_r_idx]):
                        footprint_S.add(working_routes[best_r_idx][best_p_idx])

                    working_routes[best_r_idx].insert(best_p_idx, customer)
                    current_loads[best_r_idx] += w
            elif is_mandatory:
                working_routes.append([customer])
                current_loads.append(w)
                footprint_S.add(customer)

        return [r for r in working_routes if r], len(removed_customers), list(footprint_S)
