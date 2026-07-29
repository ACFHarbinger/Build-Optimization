"""
FILO Local Search module.

Implements the localized HRVND engine with SVC-bounded move evaluation
and O(1) scaling via Selective Vertex Caching (SVC).

All operator gain computations are self-contained. Moves are generated
strictly from the SVC kernel using precomputed neighbor lists.

**O(1) State Tracking:**
- ``node_to_route_pos[u]`` → ``(route_idx, pos_idx)``  (inverse mapping)
- ``route_loads[ri]``      → current load of route ri   (incremental)

Both are maintained incrementally on every move application,
eliminating all O(N) scans from the inner evaluation loops.

The search uses Randomized Variable Neighborhood Descent (RVND):
Tier-1 operators are shuffled, each evaluated to local optimality.
If any operator improves, the tier restarts. If Tier-1 stagnates,
Tier-2 (recursive ejection chains, depth up to n_EC=3) is invoked.

Gamma sparsification probabilistically filters neighbor evaluations.

References:
    Accorsi & Vigo (2021) Section 4.4, 4.5, and 4.6.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from logic.src.policies.helpers.local_search.local_search_base import LocalSearch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_NEIGHBORS: int = 20  # Max neighbor fan-out per vertex
EJCH_MAX_DEPTH: int = 3  # Maximum ejection chain recursion depth


class FILOLocalSearch(LocalSearch):
    """Localized HRVND with O(1) state tracking, RVND, and gamma sparsification.

    **Key data structures (maintained incrementally):**

    - ``node_to_route_pos``: ``Dict[int, Tuple[int, int]]`` mapping each node
      to its ``(route_idx, position_idx)``.  Replaces O(N) ``_find_node``.
    - ``route_loads``: ``List[float]`` storing the current load of each route.
      Replaces O(N) ``sum(waste...)`` calls.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize FILO Local Search."""
        super().__init__(*args, **kwargs)
        self._node_gamma: Dict[int, float] = {}
        self.node_to_route_pos: Dict[int, Tuple[int, int]] = {}
        self._route_loads: List[float] = []

    # ------------------------------------------------------------------
    # O(1) State Management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild ``node_to_route_pos`` and ``_route_loads`` from scratch.

        Called once at the start of ``optimize()``.  Cost: O(total nodes).
        """
        self.node_to_route_pos.clear()
        self._route_loads = []
        for ri, route in enumerate(self.routes):
            load = 0.0
            for pi, node in enumerate(route):
                self.node_to_route_pos[node] = (ri, pi)
                load += self.waste.get(node, 0.0)
            self._route_loads.append(load)

    def _rebuild_route_positions(self, ri: int) -> None:
        """Rebuild position indices for a single route.  Cost: O(|route|)."""
        for pi, node in enumerate(self.routes[ri]):
            self.node_to_route_pos[node] = (ri, pi)

    def _find_node(self, u: int) -> Tuple[int, int]:
        """O(1) lookup for node position via inverse mapping."""
        return self.node_to_route_pos.get(u, (-1, -1))

    # ------------------------------------------------------------------
    # Gain functions  (self-contained, O(1) per evaluation)
    # ------------------------------------------------------------------

    def _get_gain_relocate(self, u: int, ri: int, pi: int, rj: int, pj: int) -> float:
        """Cost delta for moving *u* from route *ri* to position *pj* in *rj*."""
        route_i, route_j = self.routes[ri], self.routes[rj]
        prev_i = route_i[pi - 1] if pi > 0 else 0
        nxt_i = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        removal = self.d[prev_i, u] + self.d[u, nxt_i] - self.d[prev_i, nxt_i]

        prev_j = route_j[pj - 1] if pj > 0 else 0
        nxt_j = route_j[pj] if pj < len(route_j) else 0
        insertion = self.d[prev_j, u] + self.d[u, nxt_j] - self.d[prev_j, nxt_j]
        return (insertion - removal) * self.C

    def _get_gain_swap(self, u: int, ri: int, pi: int, v: int, rj: int, pj: int) -> float:
        """Cost delta for swapping nodes *u* and *v*."""
        if ri == rj and abs(pi - pj) == 1:
            p1, p2 = min(pi, pj), max(pi, pj)
            r = self.routes[ri]
            n1, n2 = r[p1], r[p2]
            prev1 = r[p1 - 1] if p1 > 0 else 0
            nxt2 = r[p2 + 1] if p2 < len(r) - 1 else 0
            old = self.d[prev1, n1] + self.d[n1, n2] + self.d[n2, nxt2]
            new = self.d[prev1, n2] + self.d[n2, n1] + self.d[n1, nxt2]
            return (new - old) * self.C

        route_i, route_j = self.routes[ri], self.routes[rj]
        prev_u = route_i[pi - 1] if pi > 0 else 0
        nxt_u = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        old_u = self.d[prev_u, u] + self.d[u, nxt_u]

        prev_v = route_j[pj - 1] if pj > 0 else 0
        nxt_v = route_j[pj + 1] if pj < len(route_j) - 1 else 0
        old_v = self.d[prev_v, v] + self.d[v, nxt_v]

        new_u_in_j = self.d[prev_v, u] + self.d[u, nxt_v]
        new_v_in_i = self.d[prev_u, v] + self.d[v, nxt_u]
        return (new_u_in_j + new_v_in_i - old_u - old_v) * self.C

    def _get_gain_2opt(self, ri: int, pi: int, pj: int) -> float:
        """Cost delta for reversing segment ``routes[ri][pi..pj]``."""
        if pi >= pj:
            return 0.0
        route = self.routes[ri]
        prev_i = route[pi - 1] if pi > 0 else 0
        nxt_j = route[pj + 1] if pj < len(route) - 1 else 0
        old = self.d[prev_i, route[pi]] + self.d[route[pj], nxt_j]
        new = self.d[prev_i, route[pj]] + self.d[route[pi], nxt_j]
        return (new - old) * self.C

    def _get_gain_tails(self, ri: int, pi: int, rj: int, pj: int) -> float:
        """Cost delta for swapping tails ``routes[ri][pi:]`` <-> ``routes[rj][pj:]``."""
        route_i, route_j = self.routes[ri], self.routes[rj]
        u, v = route_i[pi], route_j[pj]
        prev_u = route_i[pi - 1] if pi > 0 else 0
        prev_v = route_j[pj - 1] if pj > 0 else 0
        old = self.d[prev_u, u] + self.d[prev_v, v]
        new = self.d[prev_u, v] + self.d[prev_v, u]
        return (new - old) * self.C

    def _get_gain_split(self, ri: int, pi: int) -> float:
        """Cost delta for splitting route *ri* at position *pi*."""
        route = self.routes[ri]
        u = route[pi]
        prev_u = route[pi - 1] if pi > 0 else 0
        old = self.d[prev_u, u]
        new = self.d[prev_u, 0] + self.d[0, u]
        return (new - old) * self.C

    # ------------------------------------------------------------------
    # Gamma-gated neighbor iteration
    # ------------------------------------------------------------------

    def _gamma_neighbors(self, u: int) -> List[int]:
        """Return neighbors of *u* filtered by gamma sparsification."""
        gamma_u = self._node_gamma.get(u, 1.0)
        return [v for v in self.neighbors[u][:K_NEIGHBORS] if random.random() <= gamma_u]

    # ------------------------------------------------------------------
    # Tier-1: SVC-bounded RVND operators (O(1) lookups)
    # ------------------------------------------------------------------

    def _try_relocate(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best relocate move in the SVC kernel."""
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            w_u = self.waste.get(u, 0.0)
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1:
                    continue
                target_pj = v_pi + 1
                if ri != v_ri and self._route_loads[v_ri] + w_u > self.Q:
                    continue
                g = self._get_gain_relocate(u, ri, pi, v_ri, target_pj)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (u, ri, pi, v_ri, target_pj)
        if best_move:
            u, ri, pi, rj, pj = best_move
            w_u = self.waste.get(u, 0.0)
            self.routes[ri].pop(pi)
            self._route_loads[ri] -= w_u
            adj_pj = pj if ri != rj or pi >= pj else pj - 1
            self.routes[rj].insert(adj_pj, u)
            self._route_loads[rj] += w_u
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True
        return False

    def _try_swap(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best swap move in the SVC kernel."""
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            w_u = self.waste.get(u, 0.0)
            load_i = self._route_loads[ri]
            for v in self._gamma_neighbors(u):
                if u == v:
                    continue
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1:
                    continue
                w_v = self.waste.get(v, 0.0)
                if ri != v_ri:
                    load_j = self._route_loads[v_ri]
                    if load_i - w_u + w_v > self.Q or load_j - w_v + w_u > self.Q:
                        continue
                g = self._get_gain_swap(u, ri, pi, v, v_ri, v_pi)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (u, ri, pi, v, v_ri, v_pi)
        if best_move:
            u, ri, pi, v, v_ri, v_pi = best_move
            w_u, w_v = self.waste.get(u, 0.0), self.waste.get(v, 0.0)
            self.routes[ri][pi], self.routes[v_ri][v_pi] = v, u
            if ri != v_ri:
                self._route_loads[ri] += w_v - w_u
                self._route_loads[v_ri] += w_u - w_v
            self.node_to_route_pos[v] = (ri, pi)
            self.node_to_route_pos[u] = (v_ri, v_pi)
            return True
        return False

    def _try_2opt(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best 2-opt move in the SVC kernel."""
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1 or ri != v_ri or abs(pi - v_pi) <= 1:
                    continue
                p1, p2 = min(pi, v_pi), max(pi, v_pi)
                g = self._get_gain_2opt(ri, p1, p2)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (ri, p1, p2)
        if best_move:
            target_ri, p1, p2 = best_move
            self.routes[target_ri][p1 : p2 + 1] = reversed(self.routes[target_ri][p1 : p2 + 1])
            # Loads unchanged (intra-route); rebuild positions only
            self._rebuild_route_positions(target_ri)
            return True
        return False

    def _try_tails(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best tails (2-opt*) move in the SVC kernel."""
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            load_i = self._route_loads[ri]
            tail_i_w = sum(self.waste.get(x, 0.0) for x in self.routes[ri][pi:])
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1 or ri == v_ri:
                    continue
                load_j = self._route_loads[v_ri]
                tail_j_w = sum(self.waste.get(x, 0.0) for x in self.routes[v_ri][v_pi:])
                if load_i - tail_i_w + tail_j_w > self.Q or load_j - tail_j_w + tail_i_w > self.Q:
                    continue
                g = self._get_gain_tails(ri, pi, v_ri, v_pi)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (ri, pi, v_ri, v_pi, tail_i_w, tail_j_w)
        if best_move:
            ri, pi, rj, pj, tail_i_w, tail_j_w = best_move
            t_i, t_j = self.routes[ri][pi:], self.routes[rj][pj:]
            self.routes[ri] = self.routes[ri][:pi] + t_j
            self.routes[rj] = self.routes[rj][:pj] + t_i
            self._route_loads[ri] += tail_j_w - tail_i_w
            self._route_loads[rj] += tail_i_w - tail_j_w
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True
        return False

    def _try_split(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best split move in the SVC kernel."""
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            if random.random() > self._node_gamma.get(u, 1.0):
                continue
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            if len(self.routes[ri]) < 3 or pi == 0 or pi >= len(self.routes[ri]) - 1:
                continue
            g = self._get_gain_split(ri, pi)
            if g < best_gain - 1e-6:
                best_gain = g
                best_move = (ri, pi)
        if best_move:
            target_ri, target_pi = best_move
            new_route = self.routes[target_ri][target_pi:]
            self.routes[target_ri] = self.routes[target_ri][:target_pi]
            # Compute new route load from the split tail
            new_load = sum(self.waste.get(n, 0.0) for n in new_route)
            self._route_loads[target_ri] -= new_load
            self._route_loads.append(new_load)
            self.routes.append(new_route)
            self._rebuild_route_positions(target_ri)
            new_ri = len(self.routes) - 1
            self._rebuild_route_positions(new_ri)
            return True
        return False

    # ------------------------------------------------------------------
    # Tier-2: Recursive Ejection Chain (depth up to n_EC)
    # ------------------------------------------------------------------

    def _try_ejection_chain(self, active_nodes: Set[int]) -> bool:  # type: ignore[override]
        """Attempt a recursive ejection chain from an SVC vertex."""
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            if self._recursive_eject(u, ri, pi, depth=EJCH_MAX_DEPTH, visited=set()):
                return True
        return False

    def _recursive_eject(
        self,
        u: int,
        ri: int,
        pi: int,
        depth: int,
        visited: Set[int],
    ) -> bool:
        """Recursively eject node *u* from route *ri*, displacing others."""
        if depth <= 0:
            return False
        visited = visited | {u}
        w_u = self.waste.get(u, 0.0)

        # Compute removal savings for u
        route_i = self.routes[ri]
        prev_i = route_i[pi - 1] if pi > 0 else 0
        nxt_i = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        removal_saving = self.d[prev_i, u] + self.d[u, nxt_i] - self.d[prev_i, nxt_i]

        # Phase 1: Try direct insertion (improving)
        best_cost, best_ins = float("inf"), None
        for rj in range(len(self.routes)):
            if rj == ri:
                continue
            if self._route_loads[rj] + w_u > self.Q:
                continue
            route_j = self.routes[rj]
            for pj in range(len(route_j) + 1):
                prev = route_j[pj - 1] if pj > 0 else 0
                nxt = route_j[pj] if pj < len(route_j) else 0
                cost = self.d[prev, u] + self.d[u, nxt] - self.d[prev, nxt]
                if cost < best_cost:
                    best_cost, best_ins = cost, (rj, pj)

        if best_ins and (best_cost - removal_saving) * self.C < -1e-6:
            rj, pj = best_ins
            self.routes[ri].pop(pi)
            self._route_loads[ri] -= w_u
            self.routes[rj].insert(pj, u)
            self._route_loads[rj] += w_u
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True

        # Phase 2: Eject victim v, place u at v's position, recursively place v
        for v in self.neighbors[u][:K_NEIGHBORS]:
            if v in visited:
                continue
            v_ri, v_pi = self._find_node(v)
            if v_ri == -1 or v_ri == ri:
                continue
            w_v = self.waste.get(v, 0.0)
            if self._route_loads[v_ri] - w_v + w_u > self.Q:
                continue

            prev_v = self.routes[v_ri][v_pi - 1] if v_pi > 0 else 0
            nxt_v = self.routes[v_ri][v_pi + 1] if v_pi < len(self.routes[v_ri]) - 1 else 0
            old_v_cost = self.d[prev_v, v] + self.d[v, nxt_v]
            new_u_cost = self.d[prev_v, u] + self.d[u, nxt_v]
            local_delta = (new_u_cost - old_v_cost - removal_saving) * self.C

            if local_delta < 1e-6:
                # Speculatively apply
                self.routes[ri].pop(pi)
                self._route_loads[ri] -= w_u
                self.routes[v_ri][v_pi] = u
                self._route_loads[v_ri] += w_u - w_v
                self.node_to_route_pos[u] = (v_ri, v_pi)
                self._rebuild_route_positions(ri)

                if self._place_displaced(v, depth - 1, visited):
                    return True

                # Rollback
                self.routes[v_ri][v_pi] = v
                self._route_loads[v_ri] += w_v - w_u
                self.routes[ri].insert(pi, u)
                self._route_loads[ri] += w_u
                self.node_to_route_pos[v] = (v_ri, v_pi)
                self._rebuild_route_positions(ri)

        return False

    def _place_displaced(self, v: int, depth: int, visited: Set[int]) -> bool:
        """Try to place displaced node *v* via direct insertion or cascade."""
        w_v = self.waste.get(v, 0.0)

        # Phase A: Direct feasible insertion
        best_cost, best_ins = float("inf"), None
        for rj in range(len(self.routes)):
            if self._route_loads[rj] + w_v > self.Q:
                continue
            route_j = self.routes[rj]
            for pj in range(len(route_j) + 1):
                prev = route_j[pj - 1] if pj > 0 else 0
                nxt = route_j[pj] if pj < len(route_j) else 0
                cost = self.d[prev, v] + self.d[v, nxt] - self.d[prev, nxt]
                if cost < best_cost:
                    best_cost, best_ins = cost, (rj, pj)

        if best_ins:
            rj, pj = best_ins
            self.routes[rj].insert(pj, v)
            self._route_loads[rj] += w_v
            self._rebuild_route_positions(rj)
            return True

        # Phase B: Cascade
        if depth <= 0:
            return False

        for w in self.neighbors.get(v, [])[:K_NEIGHBORS]:
            if w in visited:
                continue
            w_ri, w_pi = self._find_node(w)
            if w_ri == -1:
                continue
            w_w = self.waste.get(w, 0.0)
            if self._route_loads[w_ri] - w_w + w_v > self.Q:
                continue

            # Speculatively eject w, insert v at w's position
            self.routes[w_ri][w_pi] = v
            self._route_loads[w_ri] += w_v - w_w
            self.node_to_route_pos[v] = (w_ri, w_pi)

            if self._place_displaced(w, depth - 1, visited | {v}):
                return True

            # Rollback
            self.routes[w_ri][w_pi] = w
            self._route_loads[w_ri] += w_w - w_v
            self.node_to_route_pos[w] = (w_ri, w_pi)

        return False

    # ------------------------------------------------------------------
    # Main optimization loop: RVND
    # ------------------------------------------------------------------

    def optimize(
        self,
        solution: List[List[int]],
        active_nodes: Optional[Set[int]] = None,
        node_gamma: Optional[List[float]] = None,
    ) -> List[List[int]]:
        """Execute RVND bounded by SVC with gamma sparsification.

        Args:
            solution: Current set of routes.
            active_nodes: Vertices in the Selective Vertex Cache.
            node_gamma: Per-node gamma values for probabilistic neighbor filtering.

        Returns:
            Improved set of routes.
        """
        self.routes = [r[:] for r in solution]
        if not active_nodes:
            return self.routes

        # Build O(1) state tracking structures
        self._rebuild_index()

        # Store gamma values
        self._node_gamma = {}
        if node_gamma:
            for i, g in enumerate(node_gamma):
                self._node_gamma[i] = g

        max_iters = getattr(self.params, "local_search_iterations", 500)
        iterations = 0

        tier1_ops = [
            ("relocate", self._try_relocate),
            ("swap", self._try_swap),
            ("2opt", self._try_2opt),
            ("tails", self._try_tails),
            ("split", self._try_split),
        ]

        improved_global = True
        while improved_global and iterations < max_iters:
            improved_global = False
            random.shuffle(tier1_ops)

            for _op_name, op_fn in tier1_ops:
                while iterations < max_iters:
                    if op_fn(active_nodes):
                        improved_global = True
                        iterations += 1
                    else:
                        break

                if improved_global:
                    break

            # Tier-2: Recursive Ejection Chain (only if Tier-1 stagnates)
            if not improved_global and iterations < max_iters and self._try_ejection_chain(active_nodes):
                improved_global = True
                iterations += 1

        return [r for r in self.routes if r]
