"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    None

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.interfaces import IAcceptanceCriterion
from logic.src.policies.helpers.operators.inter_route_local_search import (
    move_2opt_star,
    move_cross,
    move_swap_star,
    shift_2_0,
    swap_2_1,
    swap_2_2,
)
from logic.src.policies.helpers.operators.inter_route_local_search.cross_exchange import (
    cross_exchange,
    improved_cross_exchange,
    lambda_interchange,
)
from logic.src.policies.helpers.operators.inter_route_local_search.cyclic_transfer import cyclic_transfer
from logic.src.policies.helpers.operators.inter_route_local_search.ejection_chain import ejection_chain
from logic.src.policies.helpers.operators.inter_route_local_search.exchange_chain import exchange_2_0, exchange_2_1
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
)
from logic.src.policies.helpers.operators.intra_route_local_search.k_permutation import three_permutation


class LocalSearch(ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        neighbors: Optional[Dict[int, List[int]]] = None,
        penalty_capacity: float = 1.0,
        acceptance_criterion: Optional[IAcceptanceCriterion] = None,
    ):
        """
        Initialize Local Search base class.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: A parameter, likely related to route cost or penalty.
            C: A parameter, likely related to route cost or penalty.
            params: An object containing additional parameters for the local search,
                    such as time_limit.
            neighbors: Optional pre-computed neighbor map to avoid redundant sorting.
            penalty_capacity: Penalty coefficient for capacity violations (Vidal 2022).
        """
        self.d = np.array(dist_matrix)
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.penalty_capacity = penalty_capacity
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Modular Acceptance Criterion Injection
        self.acceptance_criterion = acceptance_criterion or getattr(params, "acceptance_criterion", None)
        if self.acceptance_criterion is None:
            # Fallback for base LocalSearch (elitist hill-climbing)
            from logic.src.policies.route_construction.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="only_improving")

        if neighbors is not None:
            self.neighbors = neighbors
        else:
            # Common initialization for neighbors (used by all LS)
            n_nodes = len(dist_matrix)
            self.neighbors = {}
            nb_granular = getattr(params, "nb_granular", 20)
            for i in range(1, n_nodes):
                row = self.d[i]
                order = np.argsort(row)
                cands = []
                for c in order:
                    if c not in (i, 0):
                        cands.append(c)
                        if len(cands) >= nb_granular:
                            break
                self.neighbors[i] = cands

        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []

        # Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        # Maps: node_id -> route_idx -> [(cost_delta, position), ...]
        # Stores the 3 best insertion positions for each node into each route
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}

        # Coordinate arrays for polar sector pruning (Fix 3)
        self.x_coords: Optional[np.ndarray] = None
        self.y_coords: Optional[np.ndarray] = None

        # Route sector cache: route_idx -> (min_angle, max_angle)
        self.route_sectors: Dict[int, Optional[Tuple[float, float]]] = {}

        # Current objective value tracking
        self.current_profit: float = 0.0

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """
        Optimize the given solution.
        """
        pass

    def _optimize_internal(
        self,
        target_neighborhood: Optional[str] = None,
        active_nodes: Optional[Set[int]] = None,
        initial_profit: Optional[float] = None,
    ):
        """
        Core local search loop. Assumes self.routes is populated.

        Args:
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
            active_nodes: If provided, restricts search to moves involving at least one active node.
                        This enables O(1) localization per iteration (FILO).
            initial_profit: Optional absolute profit of the starting solution.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]
        if initial_profit is not None:
            self.current_profit = initial_profit
        else:
            # Fallback slow calculation (should be avoided in tight loops)
            self.current_profit = (
                sum(self.waste.get(n, 0) * self.R for r in self.routes for n in r) - self._cost(self.routes) * self.C
            )

        # Initialize node map and sector cache
        self.node_map.clear()
        self.route_sectors.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        # Initialize Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        self._compute_top_insertions()

        # Store target neighborhood for _process_pair to filter operators
        self._target_neighborhood = target_neighborhood if target_neighborhood else "all"

        # Store active nodes set for localization (FILO)
        self._active_nodes = active_nodes

        improved = True
        it = 0
        t_start = time.process_time()
        while improved and it < self.params.local_search_iterations:
            if self.params.time_limit > 0 and time.process_time() - t_start > self.params.time_limit:
                break

            improved = False
            it += 1

            # Build the full candidate pair list
            pairs = []
            for u in self.neighbors:
                for v in self.neighbors[u]:
                    pairs.append((u, v))
            self.random.shuffle(pairs)

            for u, v in pairs:
                if self._active_nodes is not None and u not in self._active_nodes and v not in self._active_nodes:
                    continue
                if self._process_pair(u, v):
                    improved = True
                    break

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(x, 0) for x in r)

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """
        Compute or update the Top-3 Insertion Cache for O(1) SWAP* evaluation.

        Following Vidal et al. (2022), this cache stores the 3 best insertion positions
        for each node into each route, enabling constant-time insertion cost evaluation.

        Args:
            route_idx: If provided, update cache only for this route. If None, initialize for all routes.
        """
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))

        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue

            # For each node that could potentially be inserted into this route
            for node in self.neighbors:
                # Skip nodes already in this route
                if node in route:
                    continue

                # Initialize cache structure if needed
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                # Calculate insertion costs for all positions
                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.d[prev, node] + self.d[node, nxt] - self.d[prev, nxt]
                    insertion_costs.append((delta, pos))

                # Keep top 3 (lowest cost) insertions
                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance of all routes."""
        total = 0.0
        for r in routes:
            if not r:
                continue
            # Depot to first node
            total += self.d[0, r[0]]
            # Between nodes
            for i in range(len(r) - 1):
                total += self.d[r[i], r[i + 1]]
            # Last node to depot
            total += self.d[r[-1], 0]
        return total

    def _should_try_operator(self, op_name: str) -> bool:
        """Check if the operator should be tried based on target_neighborhood filter."""
        target = getattr(self, "_target_neighborhood", "all")
        return target == "all" or target == op_name

    def _polar_sector(self, route_idx: int) -> Optional[Tuple[float, float]]:
        """
        Return the (min_angle, max_angle) polar sector of a route in radians,
        measured from depot (node 0). Returns None for empty routes.
        Uses caching to avoid redundant calculations.
        """
        if route_idx in self.route_sectors:
            return self.route_sectors[route_idx]

        route = self.routes[route_idx]
        if not route:
            self.route_sectors[route_idx] = None
            return None

        x_coords = self.x_coords
        y_coords = self.y_coords
        if x_coords is None or y_coords is None:
            res = (-math.pi, math.pi)
            self.route_sectors[route_idx] = res
            return res

        angles = []
        for node in route:
            angle = math.atan2(y_coords[node] - y_coords[0], x_coords[node] - x_coords[0])
            angles.append(angle)

        res = (min(angles), max(angles))
        self.route_sectors[route_idx] = res
        return res

    def accept_move(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        """
        Consult the registered acceptance criterion to decide if a move should be applied.
        """
        if self.acceptance_criterion is None:
            return candidate_obj > current_obj + 1e-4
        accepted, _ = self.acceptance_criterion.accept(current_obj=current_obj, candidate_obj=candidate_obj, **kwargs)
        return accepted

    def step_move(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        """
        Perform internal state updates (e.g., cooling, counting) after a move decision.
        """
        if self.acceptance_criterion is not None:
            self.acceptance_criterion.step(
                current_obj=current_obj, candidate_obj=candidate_obj, accepted=accepted, **kwargs
            )

    def _sectors_overlap(self, s1: Optional[Tuple[float, float]], s2: Optional[Tuple[float, float]]) -> bool:
        """Return True if two polar sectors overlap, handling the ±π wraparound."""
        if s1 is None or s2 is None:
            return True
        # Standard interval overlap
        if s1[0] <= s2[1] and s2[0] <= s1[1]:
            return True
        # Wraparound: shift one sector by 2π and re-test
        shifted = (s1[0] + 2 * math.pi, s1[1] + 2 * math.pi)
        if shifted[0] <= s2[1] and s2[0] <= shifted[1]:
            return True
        shifted = (s2[0] + 2 * math.pi, s2[1] + 2 * math.pi)
        return bool(s1[0] <= shifted[1] and shifted[0] <= s1[1])

    def _process_pair(self, u: int, v: int) -> bool:
        """
        Attempt all applicable move operators on the (u, v) pair.
        Returns True if any improving move was applied.
        """
        u_loc = self.node_map.get(u)
        v_loc = self.node_map.get(v)

        if not u_loc and not v_loc:
            return False

        if not u_loc and v_loc:
            r_v, p_v = v_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(u, r_v, p_v)

        if u_loc and not v_loc:
            r_u, p_u = u_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(v, r_u, p_u)

        r_u, p_u = u_loc  # type: ignore[misc]
        r_v, p_v = v_loc  # type: ignore[misc]

        if r_u != r_v:
            return self._process_inter_route(u, v, r_u, p_u, r_v, p_v)
        else:
            return self._process_intra_route(u, v, r_u, p_u, r_v, p_v)

    def _process_inter_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves between different routes."""
        if (
            self._should_try_operator("relocate") or self._should_try_operator("inter_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("inter_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("inter_2opt_star") and self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("cross") and self._move_cross(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("shift_2_0") and self._move_shift_2_0(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_1") and self._move_swap_2_1(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_2") and self._move_swap_2_2(r_u, p_u, r_v, p_v):
            return True

        return (
            self._target_neighborhood == "inter_swap_star"
            and self._sectors_overlap(self._polar_sector(r_u), self._polar_sector(r_v))
            and self._move_swap_star(u, v, r_u, p_u, r_v, p_v)
        )

    def _process_intra_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves within the same route."""
        if (
            self._should_try_operator("relocate") or self._should_try_operator("intra_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("intra_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("intra_2opt") and self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("intra_or_opt") and self._move_or_opt(r_u, p_u, self.random.choice([1, 2, 3])):
            return True
        return (
            self._should_try_operator("intra_3opt")
            and getattr(self.params, "use_3opt", False)
            and self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v, self.random)
        )

    def _update_map(self, affected_indices: Set[int]):
        """
        Update node map and route loads after route modifications.
        Also updates Top-3 Insertion Cache for affected routes (Vidal 2022).

        Args:
            affected_indices: Set of route indices that were modified.
        """
        for ri in sorted(affected_indices):
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
            # Invalidate and update Top-3 cache for this modified route (O(1) SWAP* maintenance)
            self.route_sectors.pop(ri, None)
            self._compute_top_insertions(route_idx=ri)

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, r_idx: int, pos: int, chain_len: int) -> bool:
        return move_or_opt(self, r_idx, pos, chain_len)

    def _move_cross(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_cross(self, u, v, r_u, p_u, r_v, p_v)

    def _move_shift_2_0(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return shift_2_0(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_1(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return swap_2_1(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_2(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return swap_2_2(self, r_u, p_u, r_v, p_v)

    def _try_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_improved_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return improved_cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_lambda_interchange(self, r_u: int, r_v: int) -> bool:
        return lambda_interchange(self, getattr(self.params, "lambda_max", 2))

    def _try_cyclic_transfer(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if len(self.routes) < 3:
            return False
        candidates = [r for r in range(len(self.routes)) if r != r_u and r != r_v and self.routes[r]]
        if not candidates:
            return False
        r_w = self.random.choice(candidates)
        p_w = self.random.randint(0, len(self.routes[r_w]) - 1)
        return cyclic_transfer(self, [(r_u, p_u), (r_v, p_v), (r_w, p_w)])

    def _try_exchange_chains(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if exchange_2_0(self, r_u, p_u, r_v, p_v):
            return True
        return bool(exchange_2_1(self, r_u, p_u, r_v, p_v))

    def _try_ejection_chain(self, r_u: int) -> bool:
        return ejection_chain(self, r_u)

    def _move_relocate_chain(self, u: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=2):
            return True
        return bool(relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=3))

    def _move_three_permutation(self, u: int, r_u: int, p_u: int) -> bool:
        return three_permutation(self, r_u, p_u)

    def _move_unrouted_insert(self, node: int, route_idx: int, _hint_pos: int) -> bool:
        """
        Try inserting an unrouted node back into a route in O(1) time (VRPP operator).

        Evaluates inserting 'node' (currently unvisited) into 'route_idx' at the best position.
        Uses Top-3 Insertion Cache for O(1) position lookup instead of O(N) loop.
        Uses penalized search: considers both profit gain and capacity penalties.

        Args:
            node: Unvisited node to potentially insert.
            route_idx: Route to insert into.
            _hint_pos: Position hint (near a neighbor), currently unused.

        Returns:
            bool: True if insertion improves penalized profit, False otherwise.
        """
        route = self.routes[route_idx]

        # 1. Base Case: Empty route
        if not route:
            cost_delta = 2 * self.d[0, node] * self.C
            best_pos = 0
        else:
            # 2. O(1) Cache Query - No loops required!
            # Since 'node' is unrouted, all cached positions are valid (no removal offsets)
            if node in self.top_insertions and route_idx in self.top_insertions[node]:
                # Always use first cached position (guaranteed valid for unrouted nodes)
                best_delta_raw, best_pos = self.top_insertions[node][route_idx][0]
                cost_delta = best_delta_raw * self.C
            else:
                # Fallback safety if cache is missing (shouldn't happen with proper maintenance)
                return False

        profit_delta = self.waste.get(node, 0) * self.R
        load_delta = self.waste.get(node, 0)

        # 3. Penalties & Evaluation (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # Accept if improves penalized profit
        if total_change > 1e-4:
            if not route:
                self.routes[route_idx] = [node]
            else:
                self.routes[route_idx].insert(best_pos, node)
            self._update_map({route_idx})
            return True

        return False
