"""
Split algorithm for Hybrid Genetic Search (HGS).

This module implements the linear-time Split algorithm used to partition
a giant tour into optimal routes based on vehicle capacity.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class LinearSplit:
    """
    Linear-time Split algorithm for decoding giant tours into routes.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        max_vehicles: int = 0,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the LinearSplit solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            max_vehicles: Maximum number of vehicles allowed (0 for unlimited).
            mandatory_nodes: List of local node indices that MUST be visited.
        """
        self.dist_matrix = np.array(dist_matrix)
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.max_vehicles = max_vehicles
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()

    def split(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """
        Partition a giant tour into feasible routes.

        Args:
            giant_tour: Ordered list of all client nodes to be visited.

        Returns:
            Tuple[List[List[int]], float]: List of routes and total profit.
        """
        if not giant_tour:
            return [], 0.0

        n = len(giant_tour)

        cum_load = [0.0] * (n + 1)
        cum_rev = [0.0] * (n + 1)
        cum_dist = [0.0] * (n + 1)

        dmat = self.dist_matrix
        wastes = self.wastes
        R_val = self.R

        load_curr = 0.0
        rev_curr = 0.0
        dist_curr = 0.0
        prev_node = 0

        d_0_x = [0.0] * (n + 1)
        d_x_0 = [0.0] * (n + 1)

        for i in range(1, n + 1):
            node = giant_tour[i - 1]
            dem = wastes.get(node, 0)

            load_curr += dem
            rev_curr += dem * R_val

            if i > 1:
                dist_curr += dmat[prev_node, node]

            cum_load[i] = load_curr
            cum_rev[i] = rev_curr
            cum_dist[i] = dist_curr

            d_0_x[i] = dmat[0, node]
            d_x_0[i] = dmat[node, 0]
            prev_node = node

        res: Tuple[List[List[int]], float] = ([], -float("inf"))
        if self.max_vehicles == 0:
            res = self._split_unlimited(n, giant_tour, cum_load, cum_rev, cum_dist, d_0_x, d_x_0)
        else:
            res = self._split_limited(n, giant_tour, cum_load, cum_rev, cum_dist, d_0_x, d_x_0)

        if res[1] == -float("inf"):
            return self._fallback_split(giant_tour)

        return res

    def _fallback_split(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        routes = []
        current_route = []
        current_load = 0.0

        for node in giant_tour:
            dem = self.wastes.get(node, 0)
            if current_load + dem <= self.capacity:
                current_route.append(node)
                current_load += dem
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [node]
                current_load = dem

        if current_route:
            routes.append(current_route)

        rev = sum(self.wastes.get(n, 0) for r in routes for n in r) * self.R
        cost = 0
        for r in routes:
            d = self.dist_matrix[0, r[0]]
            for k in range(len(r) - 1):
                d += self.dist_matrix[r[k], r[k + 1]]
            d += self.dist_matrix[r[-1], 0]
            cost += d * self.C

        profit = rev - cost
        return routes, profit

    def _split_unlimited(
        self,
        n: int,
        nodes: List[int],
        cum_load: List[float],
        cum_rev: List[float],
        cum_dist: List[float],
        d_0_x: List[float],
        d_x_0: List[float],
    ) -> Tuple[List[List[int]], float]:
        V = [-float("inf")] * (n + 1)
        P = [-1] * (n + 1)
        V[0] = 0.0

        C_cost = self.C
        cap = self.capacity

        term_0 = V[0] - cum_rev[0] + C_cost * (cum_dist[1] - d_0_x[1])
        dq = deque([(0, term_0)])

        for i in range(1, n + 1):
            min_load = cum_load[i] - cap
            while dq:
                idx = dq[0][0]
                if cum_load[idx] < min_load - 1e-5:
                    dq.popleft()
                else:
                    break

            if dq:
                best_j, best_A = dq[0]
                B_i = cum_rev[i] - C_cost * (cum_dist[i] + d_x_0[i])
                V[i] = best_A + B_i
                P[i] = best_j

            # 2. Skip transition (only if NOT mandatory)
            node = nodes[i - 1]
            if node not in self.mandatory_nodes and V[i - 1] > V[i]:
                V[i] = V[i - 1]
                P[i] = -2  # Marker for skip

            if i < n:
                j_new = i
                if V[j_new] > -float("inf"):
                    idx_next = i + 1
                    term = V[j_new] - cum_rev[j_new] + C_cost * (cum_dist[idx_next] - d_0_x[idx_next])
                    while dq:
                        if dq[-1][1] <= term:
                            dq.pop()
                        else:
                            break
                    dq.append((j_new, term))

        return self._reconstruct(n, nodes, P, V[n])

    def _split_limited(
        self,
        n: int,
        nodes: List[int],
        cum_load: List[float],
        cum_rev: List[float],
        cum_dist: List[float],
        d_0_x: List[float],
        d_x_0: List[float],
    ) -> Tuple[List[List[int]], float]:
        K = self.max_vehicles
        V_prev = [-float("inf")] * (n + 1)
        V_prev[0] = 0.0

        P = [[-1] * (n + 1) for _ in range(K + 1)]
        best_profit = -float("inf")
        best_k = 0

        C_cost = self.C
        cap = self.capacity

        for k in range(1, K + 1):
            V_curr = [-float("inf")] * (n + 1)
            dq: Deque[Tuple[int, float]] = deque()

            if V_prev[0] > -float("inf"):
                term_0 = V_prev[0] - cum_rev[0] + C_cost * (cum_dist[1] - d_0_x[1])
                dq.append((0, term_0))

            for i in range(1, n + 1):
                min_load = cum_load[i] - cap
                while dq:
                    idx = dq[0][0]
                    if cum_load[idx] < min_load - 1e-5:
                        dq.popleft()
                    else:
                        break

                if dq:
                    best_j, best_A = dq[0]
                    B_i = cum_rev[i] - C_cost * (cum_dist[i] + d_x_0[i])
                    V_curr[i] = best_A + B_i
                    P[k][i] = best_j

                # 2. Skip transition (only if NOT mandatory)
                node = nodes[i - 1]
                if node not in self.mandatory_nodes and V_curr[i - 1] > V_curr[i]:
                    V_curr[i] = V_curr[i - 1]
                    P[k][i] = -2  # Marker for skip

                if i < n:
                    j_new = i
                    if V_prev[j_new] > -float("inf"):
                        idx_next = i + 1
                        term = V_prev[j_new] - cum_rev[j_new] + C_cost * (cum_dist[idx_next] - d_0_x[idx_next])
                        while dq:
                            if dq[-1][1] <= term:
                                dq.pop()
                            else:
                                break
                        dq.append((j_new, term))

            if V_curr[n] > best_profit:
                best_profit = V_curr[n]
                best_k = k

            V_prev = V_curr

        if best_profit == -float("inf"):
            return [], -float("inf")

        return self._reconstruct_limited(n, nodes, P, best_k, best_profit)

    def _reconstruct_limited(
        self,
        n: int,
        nodes: List[int],
        P: List[List[int]],
        k_opt: int,
        total_profit: float,
    ) -> Tuple[List[List[int]], float]:
        routes = []
        curr = n
        k = k_opt

        while curr > 0 and k > 0:
            prev = P[k][curr]
            if prev == -1:
                return [], -float("inf")
            if prev == -2:
                # Skip current node
                curr -= 1
                continue
            segment = nodes[prev:curr]
            routes.append(segment)
            curr = prev
            k -= 1

        if curr != 0:
            # Handle remaining skipped nodes if any
            while curr > 0 and P[k][curr] == -2:
                curr -= 1

        if curr != 0:
            return [], -float("inf")

        return routes[::-1], total_profit

    def _reconstruct(
        self, n: int, nodes: List[int], P: List[int], total_profit: float
    ) -> Tuple[List[List[int]], float]:
        if total_profit == -float("inf"):
            return [], -float("inf")
        routes = []
        curr = n
        while curr > 0:
            prev = P[curr]
            if prev == -1:
                return [], -float("inf")
            if prev == -2:
                # Skip current node
                curr -= 1
                continue
            routes.append(nodes[prev:curr])
            curr = prev
        routes.reverse()
        return routes, total_profit


def split_algorithm(
    giant_tour: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
):
    """
    Convenience wrapper for the LinearSplit algorithm.

    Args:
        giant_tour: Giant tour to be split.
        dist_matrix: Distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Configuration parameters including `max_vehicles`.
        mandatory_nodes: List of mandatory node indices.

    Returns:
        Tuple[List[List[int]], float]: Decoded routes and total profit.
    """
    s = LinearSplit(dist_matrix, wastes, capacity, R, C, values.get("max_vehicles", 0), mandatory_nodes)
    return s.split(giant_tour)
