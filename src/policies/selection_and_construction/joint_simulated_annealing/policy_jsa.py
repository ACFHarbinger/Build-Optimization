"""
Joint Simulated Annealing Policy.

A meta-heuristic policy for the Traveling Thief Problem / Thief Orienteering Problem
style combined selection and routing.
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.jsa import JointSAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.selection_and_construction.base.base_joint_policy import BaseJointPolicy
from logic.src.policies.selection_and_construction.base.registry import JointPolicyRegistry

from .params import JointSAParams


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.SELECTION,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.SINGLE_PERIOD,
    PolicyTag.JOINT,
)
@JointPolicyRegistry.register("jsa")
@RouteConstructorRegistry.register("jsa")
class JointSAPolicy(BaseJointPolicy):
    """
    Simulated Annealing policy for joint mandatory-bin selection and route construction.

    State space:
        - Selection: Bitstring (mask) of bins to visit.
        - Routing: Permutation of selected bins (tours).

    Objective:
        Maximise Net Profit (Revenue - Distance Cost) - Overflow Penalties.
    """

    def __init__(self, config: Optional[Union[JointSAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return JointSAConfig

    def _get_config_key(self) -> str:
        return "jsa"

    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """
        Execute SA loop to find near-optimal selection and routing.
        """
        params = JointSAParams.from_config(self._config) if self._config else JointSAParams()

        start_time = time.time()
        random.seed(params.seed)
        np.random.seed(params.seed)

        n_bins = len(context.bin_ids)
        if n_bins == 0:
            return [], [], 0.0, 0.0

        # 1. Initial State: Greedy construction
        current_selection = self._initial_selection(context)
        current_routes = self._construct_greedy_routes(current_selection, context)
        current_score, current_profit, current_cost = self._evaluate(current_selection, current_routes, context, params)

        best_selection = list(current_selection)
        best_routes = [list(r) for r in current_routes]
        best_score = current_score

        # 2. SA Loop
        temp = params.start_temp
        step = 0

        while step < params.max_steps and (time.time() - start_time) < params.time_limit:
            step += 1

            # 3. Generate Neighbor
            neighbor_selection, neighbor_routes = self._generate_neighbor(
                current_selection, current_routes, context, params
            )
            neighbor_score, neighbor_profit, neighbor_cost = self._evaluate(
                neighbor_selection, neighbor_routes, context, params
            )

            # 4. Acceptance
            delta = neighbor_score - current_score
            if delta > 0 or (temp > 0 and random.random() < np.exp(delta / temp)):
                current_selection = neighbor_selection
                current_routes = neighbor_routes
                current_score = neighbor_score
                current_profit = neighbor_profit
                current_cost = neighbor_cost

                # Update best
                if current_score > best_score:
                    best_selection = list(current_selection)
                    best_routes = [list(r) for r in current_routes]
                    best_score = current_score

            # 5. Cool
            temp *= params.cooling_rate
            if temp < 1e-8:
                temp = params.start_temp * 0.1  # Minimal reheat

        return best_selection, best_routes, current_profit, current_cost

    def _initial_selection(self, context: JointSelectionConstructionContext) -> List[int]:
        """Simple initial selection: bins > 80% full."""
        return [bid for i, bid in enumerate(context.bin_ids) if context.current_fill[i] > 80.0]

    def _construct_greedy_routes(
        self, selected_bins: List[int], context: JointSelectionConstructionContext
    ) -> List[List[int]]:
        """Greedy insertion routing for a fixed selection."""
        if not selected_bins:
            return []

        sub_dist, sub_wastes, _ = self._build_subset_problem(selected_bins, context)
        n_selected = len(selected_bins)
        unvisited = list(range(1, n_selected + 1))

        routes: List[List[int]] = []
        while unvisited:
            route = []
            curr_cap = context.capacity
            curr_loc = 0  # Depot

            while unvisited:
                # Find nearest neighbor that fits
                best_next = -1
                best_dist = float("inf")

                for node in unvisited:
                    mass = sub_wastes[node] * (context.bin_density * context.bin_volume / 100.0)
                    if mass <= curr_cap:
                        d = sub_dist[curr_loc, node]
                        if d < best_dist:
                            best_dist = d
                            best_next = node

                if best_next != -1:
                    route.append(best_next)
                    mass = sub_wastes[best_next] * (context.bin_density * context.bin_volume / 100.0)
                    curr_cap -= mass
                    curr_loc = best_next
                    unvisited.remove(best_next)
                else:
                    break  # Tour full

            if route:
                routes.append(route)
            else:
                break

        return routes

    def _evaluate(
        self,
        selected_bins: List[int],
        routes: List[List[int]],
        context: JointSelectionConstructionContext,
        params: JointSAParams,
    ) -> Tuple[float, float, float]:
        """Calculate score (Profit - Penalty)."""
        if not routes:
            return -self._total_overflow_penalty(selected_bins, context, params), 0.0, 0.0

        # 1. Total Profit
        total_mass = 0.0
        for bid in selected_bins:
            idx = np.where(context.bin_ids == bid)[0][0]
            total_mass += context.current_fill[idx] * (context.bin_density * context.bin_volume / 100.0)

        revenue = total_mass * context.revenue_kg

        _, _, subset_indices = self._build_subset_problem(selected_bins, context)
        tour = self._routes_to_flat_tour(routes, subset_indices)

        dm = np.asarray(context.distance_matrix, dtype=float)
        dist_cost = 0.0
        for i in range(len(tour) - 1):
            dist_cost += dm[tour[i], tour[i + 1]]

        profit = revenue - (dist_cost * context.cost_per_km)

        # 2. Penalties
        overflow_penalty = self._total_overflow_penalty(selected_bins, context, params)

        score = profit - overflow_penalty
        return score, profit, dist_cost

    def _total_overflow_penalty(
        self, selected_bins: List[int], context: JointSelectionConstructionContext, params: JointSAParams
    ) -> float:
        """Penalty for bins NOT selected that are at risk."""
        selected_set = set(selected_bins)
        penalty = 0.0

        for i, bid in enumerate(context.bin_ids):
            if bid not in selected_set:
                fill = context.current_fill[i]
                if fill > 90.0:
                    penalty += params.overflow_penalty * (fill / 100.0)

        return penalty

    def _generate_neighbor(
        self,
        current_selection: List[int],
        current_routes: List[List[int]],
        context: JointSelectionConstructionContext,
        params: JointSAParams,
    ) -> Tuple[List[int], List[List[int]]]:
        """Move: Flip selection OR swap routing."""
        new_selection = list(current_selection)
        new_routes = [list(r) for r in current_routes]

        if random.random() < params.prob_bit_flip or not current_selection:
            node_to_flip = int(random.choice(context.bin_ids))
            if node_to_flip in new_selection:
                new_selection.remove(node_to_flip)
            else:
                new_selection.append(node_to_flip)

            new_routes = self._construct_greedy_routes(new_selection, context)
        else:
            if len(new_routes) >= 1:
                r_idx = random.randrange(len(new_routes))
                if len(new_routes[r_idx]) >= 2:
                    i, j = random.sample(range(len(new_routes[r_idx])), 2)
                    new_routes[r_idx][i], new_routes[r_idx][j] = new_routes[r_idx][j], new_routes[r_idx][i]

        return new_selection, new_routes
