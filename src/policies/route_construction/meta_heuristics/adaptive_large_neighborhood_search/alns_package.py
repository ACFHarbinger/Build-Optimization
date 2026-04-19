"""
Integration with the external `alns` Python package.

This module provides the necessary state representation and operator wrappers
to use the `alns` package as a solver backend for the routing problem.
"""

import copy
from typing import Any, Dict, List, Optional, cast

import numpy as np
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

from logic.src.tracking.viz_mixin import PolicyStateRecorder


class ALNSState:
    """
    State representation for the `alns` Python package.

    Encapsulates the current solution (routes and unassigned nodes) and
    provides methods for copying and objective evaluation.
    """

    def __init__(
        self,
        routes: List[List[int]],
        unassigned: List[int],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        values: Dict[str, Any],
    ):
        """
        Initialize the ALNS state.

        Args:
            routes: List of routes, where each route is a list of node indices.
            unassigned: List of node indices not yet assigned to any route.
            dist_matrix: NxN distance matrix.
            wastes: Dictionary mapping node indices to their waste/weight.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            values: Additional configuration parameters.
        """
        self.routes = routes
        self.unassigned = unassigned
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.values = values
        self._score: float = self.calculate_profit()

    def copy(self) -> "ALNSState":
        """
        Create a deep copy of the current state.

        Returns:
            ALNSState: A new state object with copied routes and unassigned nodes.
        """
        return ALNSState(
            copy.deepcopy(self.routes),
            copy.deepcopy(self.unassigned),
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            self.values,
        )

    def objective(self) -> float:
        """
        Calculate the objective value for the `alns` package (minimization).

        Returns:
            float: The negative profit (since ALNS minimizes the objective).
        """
        return -self._score

    def calculate_profit(self) -> float:
        """
        Calculate the total profit of the current solution.

        Returns:
            float: Revenue minus routing costs.
        """
        total_profit = 0.0
        for route in self.routes:
            if not route:
                continue

            dist = 0.0
            load = 0.0
            dist += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
                load += self.wastes[route[i]]
            dist += self.dist_matrix[route[-1]][0]
            load += self.wastes[route[-1]]

            cost = dist * self.C
            revenue = sum(self.wastes[node] * self.R for node in route)
            total_profit += revenue - cost

        return total_profit

    @property
    def cost(self) -> float:
        """
        Alias for objective() to satisfy external package expectations.

        Returns:
            float: The objective value.
        """
        return self.objective()


def alns_pkg_random_removal(state: ALNSState, rng: np.random.Generator, **kwargs: Any) -> ALNSState:
    """
    Randomly remove nodes from the current solution for the `alns` package.

    Args:
        state: The current ALNSState.
        rng: NumPy Generator for reproducibility.

    Returns:
        ALNSState: A new state with nodes removed.
    """
    new_state = state.copy()
    all_nodes = [n for r in new_state.routes for n in r]
    if not all_nodes:
        return new_state
    n_remove = min(len(all_nodes), int(rng.integers(1, min(len(all_nodes) + 1, 10))))
    removed = rng.choice(all_nodes, n_remove, replace=False)

    new_routes = []
    for r in new_state.routes:
        new_r = [n for n in r if n not in removed]
        if new_r:
            new_routes.append(new_r)

    new_state.routes = new_routes
    new_state.unassigned.extend(removed)
    new_state._score = new_state.calculate_profit()
    return new_state


def alns_pkg_worst_removal(state: ALNSState, rng: np.random.Generator, **kwargs: Any) -> ALNSState:
    """
    Remove nodes with the highest cost contribution for the `alns` package.

    Args:
        state: The current ALNSState.
        rng: NumPy Generator for reproducibility.

    Returns:
        ALNSState: A new state with nodes removed.
    """
    new_state = state.copy()
    all_nodes = [n for r in new_state.routes for n in r]
    if not all_nodes:
        return new_state

    costs = []
    for _r_idx, route in enumerate(new_state.routes):
        for idx, node in enumerate(route):
            prev_node = 0 if idx == 0 else route[idx - 1]
            next_node = 0 if idx == len(route) - 1 else route[idx + 1]
            cost = (
                state.dist_matrix[prev_node][node]
                + state.dist_matrix[node][next_node]
                - state.dist_matrix[prev_node][next_node]
            )
            costs.append((cost, node))

    costs.sort(key=lambda x: x[0], reverse=True)
    n_remove = min(len(all_nodes), int(rng.integers(1, min(len(all_nodes) + 1, 10))))
    removed = [x[1] for x in costs[:n_remove]]

    new_routes = []
    for r in new_state.routes:
        new_r = [n for n in r if n not in removed]
        if new_r:
            new_routes.append(new_r)

    new_state.routes = new_routes
    new_state.unassigned.extend(removed)
    new_state._score = new_state.calculate_profit()
    return new_state


def alns_pkg_greedy_insertion(state: ALNSState, rng: np.random.Generator, **kwargs: Any) -> ALNSState:
    """
    Greedily insert unassigned nodes into the best positions for the `alns` package.

    Args:
        state: The current ALNSState.
        rng: NumPy Generator for reproducibility.

    Returns:
        ALNSState: A new state with nodes re-inserted.
    """
    new_state = state.copy()
    rng.shuffle(new_state.unassigned)
    while new_state.unassigned:
        node = new_state.unassigned.pop(0)
        best_cost = float("inf")
        best_pos = None
        candidate_routes = new_state.routes + [[]]
        for r_idx, route in enumerate(candidate_routes):
            current_load = sum(state.wastes[n] for n in route)
            if current_load + state.wastes[node] > state.capacity:
                continue
            for i in range(len(route) + 1):
                prev_node = 0 if i == 0 else route[i - 1]
                next_node = 0 if i == len(route) else route[i]
                delta_dist = (
                    state.dist_matrix[prev_node][node]
                    + state.dist_matrix[node][next_node]
                    - state.dist_matrix[prev_node][next_node]
                )
                delta_leverage = (delta_dist * state.C) - (state.wastes[node] * state.R)
                if delta_leverage < best_cost:
                    best_cost = delta_leverage
                    best_pos = (r_idx, i)
        if best_pos:
            r_idx, idx = best_pos
            if r_idx >= len(new_state.routes):
                new_state.routes.append([node])
            else:
                new_state.routes[r_idx].insert(idx, node)
    new_state._score = new_state.calculate_profit()
    return new_state


def run_alns_package(dist_matrix, wastes, capacity, R, C, values, recorder: Optional[PolicyStateRecorder] = None):
    """
    Execute the ALNS algorithm using the external `alns` package.

    Args:
        dist_matrix: NxN distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Configuration parameters including `time_limit`.

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, total profit, and total distance cost.
    """
    n_nodes = len(dist_matrix) - 1
    nodes = list(range(1, n_nodes + 1))

    initial_routes = [[i] for i in nodes]
    init_state = ALNSState(initial_routes, [], dist_matrix, wastes, capacity, R, C, values)

    seed = values.get("seed", 42)
    alns = ALNS(np.random.default_rng(seed))
    alns.add_destroy_operator(alns_pkg_random_removal)  # pyrefly: ignore[bad-argument-type]
    alns.add_destroy_operator(alns_pkg_worst_removal)  # pyrefly: ignore[bad-argument-type]
    alns.add_repair_operator(alns_pkg_greedy_insertion)  # pyrefly: ignore[bad-argument-type]

    time_limit = values.get("time_limit", 10)
    select = RouletteWheel([25, 10, 1, 0], 0.8, 1, 1)
    accept = SimulatedAnnealing(start_temperature=1000, end_temperature=1, step=1 - 1e-3)
    stop = MaxRuntime(time_limit)

    result = alns.iterate(init_state, select, accept, stop, collect_stats=True)
    best_state: ALNSState = cast(ALNSState, result.best_state)

    if recorder is not None:
        try:
            for step, obj in enumerate(result.statistics.objectives):
                recorder.record(iteration=step, best_profit=-obj)
        except Exception:  # noqa: BLE001
            pass

    total_dist_cost = 0
    for route in best_state.routes:
        if not route:
            continue
        d = dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            d += dist_matrix[route[i]][route[i + 1]]
        d += dist_matrix[route[-1]][0]
        total_dist_cost += d * C

    profit = best_state.calculate_profit()
    return best_state.routes, profit, total_dist_cost
