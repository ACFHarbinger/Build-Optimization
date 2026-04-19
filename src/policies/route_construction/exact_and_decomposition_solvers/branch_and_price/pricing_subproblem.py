"""
Pricing Subproblem for Branch-and-Price VRPP.

Solves the resource-constrained shortest path problem (RCSPP) heuristically
to generate new routes with positive reduced cost.  The pricing problem finds
routes that improve the LP relaxation of the master problem.

Based on Section 2.2 and Section 4 of Barnhart et al. (1998).

Branching constraint enforcement
---------------------------------
Active branching constraints are now passed into every route-generation call
and are enforced eagerly during the greedy insertion loop — not post-hoc.
This prevents the heuristic from producing columns that violate the current
B&B node's branching decisions, which would pollute the master LP and cause
the LP bound to be over-estimated.

Both constraint flavours defined in ``branching.py`` are handled:

EdgeBranchingConstraint (must_use=False)
    Forbids arc (u → v).  Before inserting node v after u, check forbidden
    arcs.

EdgeBranchingConstraint (must_use=True)
    Requires arc (u → w).  If the current tail of the partial route is u and
    the candidate node is v ≠ w, reject v (u must be followed by w next).
    Symmetrically, if v has a required predecessor x and the current tail is
    not x, reject v.

RyanFosterBranchingConstraint (together=False)
    Forbids r and s from appearing in the same route.  If v is already in a
    separate constraint with a node already in the partial route, reject v.

RyanFosterBranchingConstraint (together=True)
    Best-effort at heuristic level: enforced lazily at route completion by
    discarding any finished route that contains exactly one node of a
    together-pair (i.e. partial together enforcement).  Full enforcement
    would require a DP.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class PricingSubproblem:
    """
    Resource-Constrained Shortest Path Problem for Route Generation (Heuristic).

    Uses a greedy insertion heuristic to quickly generate routes with positive
    reduced cost.  For exact pricing, use :class:`rcspp_dp.RCSPPSolver`.

    The heuristic respects active branching constraints during construction so
    that generated columns are always feasible at the current B&B node.
    """

    IS_EXACT = False  # Heuristic pricer — must never be used for LP optimality proofs.

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Set[int],
    ) -> None:
        """
        Initialise the pricing subproblem.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = 0).
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
            wastes: Mapping from node ID to waste volume.
            capacity: Vehicle payload capacity.
            revenue_per_kg: Revenue earned per unit of waste collected.
            cost_per_km: Cost per unit of distance travelled.
            mandatory_nodes: Set of node indices that must be visited.
            params: Standardized BP parameters.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes
        self.depot = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        dual_values: Dict[Any, Any],
        max_routes: int = 10,
        active_constraints: Optional[List[Any]] = None,
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        assert_not_exact: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[List[int], float]]:
        """
        Generate routes with positive reduced cost.

        **Warning:** This pricer is heuristic. It may miss columns with positive reduced
        cost. It must only be used for warm-starting the initial column pool, never as the
        pricer inside the B&B column generation loop where LP optimality must be certified.

        Tries multiple starting nodes to diversify the column pool.  Each
        candidate route is built with the greedy insertion heuristic and is
        guaranteed to satisfy all active branching constraints.

        Args:
            dual_values: Dual values from the master problem.  Keys are
                customer node IDs (int) and optionally ``"vehicle_limit"``.
            max_routes: Maximum number of routes to return.
            active_constraints: Active branching constraints at the current
                B&B node, or ``None`` / empty list for the root.  Accepts
                both :class:`branching.EdgeBranchingConstraint` and
                :class:`branching.RyanFosterBranchingConstraint` instances.
            capacity_cut_duals: Optional mapping of node-sets to dual values
                for active capacity cuts.
            assert_not_exact: If True, raises RuntimeError to prevent misuse
                in an exact B&B pricing loop.

        Returns:
            List of ``(route_nodes, reduced_cost)`` tuples sorted by
            descending reduced cost.  ``route_nodes`` excludes the depot.
        """
        if assert_not_exact:
            raise RuntimeError(
                "PricingSubproblem is a heuristic and cannot be used to certify LP "
                "optimality at a B&B node. Use RCSPPSolver instead."
            )

        # Support both flat node duals (legacy) and structured dual dicts (BPC/standardized)
        node_duals: Dict[Any, float]
        if "node_duals" in dual_values:
            node_duals = dual_values["node_duals"]
            if capacity_cut_duals is None:
                capacity_cut_duals = dual_values.get("rcc_duals")
        else:
            node_duals = dual_values  # type: ignore[assignment]

        constraints: List[Any] = active_constraints or []

        # Pre-process constraints into O(1) look-up structures.
        (
            forbidden_arcs,
            req_successors,
            req_predecessors,
            rf_separate_pairs,
            rf_together_pairs,
        ) = self._preprocess_constraints(constraints)

        routes: List[Tuple[List[int], float]] = []

        # Rank start nodes by their standalone attractiveness.
        candidate_starts = sorted(
            range(1, self.n_nodes + 1),
            key=lambda n: node_duals.get(n, 0.0) + self.wastes.get(n, 0.0) * self.R,
            reverse=True,
        )

        for start_node in candidate_starts[:max_routes]:
            route, rc = self._greedy_route_construction(
                start_node=start_node,
                dual_values=node_duals,
                forbidden_arcs=forbidden_arcs,
                req_successors=req_successors,
                req_predecessors=req_predecessors,
                rf_separate_pairs=rf_separate_pairs,
                rf_together_pairs=rf_together_pairs,
                capacity_cut_duals=capacity_cut_duals,
                node_prizes=kwargs.get("node_prizes"),
            )

            if rc > 1e-4:
                routes.append((route, rc))

            if len(routes) >= max_routes:
                break

        routes.sort(key=lambda x: x[1], reverse=True)
        return routes[:max_routes]

    # ------------------------------------------------------------------
    # Constraint pre-processing
    # ------------------------------------------------------------------

    def _preprocess_constraints(
        self,
        constraints: List[Any],
    ) -> Tuple[
        Set[Tuple[int, int]],  # forbidden_arcs
        Dict[int, int],  # req_successors:  u → required next node
        Dict[int, int],  # req_predecessors: v → required previous node
        Set[Tuple[int, int]],  # rf_separate_pairs: frozenset-like sorted pairs
        Set[Tuple[int, int]],  # rf_together_pairs: sorted pairs
    ]:
        """
        Convert active constraints into fast look-up structures.

        Separates EdgeBranchingConstraint objects from RyanFosterBranchingConstraint
        objects without importing the classes (uses duck-typing to stay
        decoupled from the branching module at import time).

        Args:
            constraints: Mixed list of active branching constraints.

        Returns:
            Five look-up structures used by the greedy heuristic.
        """
        forbidden_arcs: Set[Tuple[int, int]] = set()
        req_successors: Dict[int, int] = {}
        req_predecessors: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()

        for c in constraints:
            # Duck-typed dispatch — avoids a hard import dependency on branching.py.
            if hasattr(c, "must_use"):
                # EdgeBranchingConstraint
                if not c.must_use:
                    forbidden_arcs.add((c.u, c.v))
                else:
                    req_successors[c.u] = c.v
                    req_predecessors[c.v] = c.u
            elif hasattr(c, "together"):
                # RyanFosterBranchingConstraint
                key = (min(c.node_r, c.node_s), max(c.node_r, c.node_s))
                if not c.together:
                    rf_separate.add(key)
                else:
                    rf_together.add(key)

        return forbidden_arcs, req_successors, req_predecessors, rf_separate, rf_together

    # ------------------------------------------------------------------
    # Greedy route construction
    # ------------------------------------------------------------------

    def _evaluate_candidate_insertion(
        self,
        v: int,
        v_waste: float,
        route: List[int],
        dual_values: Dict[Any, float],
        forbidden_arcs: Set[Tuple[int, int]],
        req_successors: Dict[int, int],
        req_predecessors: Dict[int, int],
        rf_separate_pairs: Set[Tuple[int, int]],
        visited: Set[int],
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> Tuple[Optional[int], float]:
        """Evaluate candidate v and find best insertion position and profit."""
        # Rule 4: Ryan-Foster separation.
        if self._violates_rf_separation(v, visited, rf_separate_pairs):
            return None, -float("inf")

        best_pos = None
        best_marginal = -float("inf")

        # ---- Marginal profit (best insertion position) ----------
        for pos in range(len(route) + 1):
            prev = self.depot if pos == 0 else route[pos - 1]
            nxt = self.depot if pos == len(route) else route[pos]

            # Rule 1: Forbidden arcs.
            if (prev, v) in forbidden_arcs or (v, nxt) in forbidden_arcs:
                continue

            # Rule 2: Required successors.
            if prev in req_successors and req_successors[prev] != v:
                continue
            if v in req_successors and req_successors[v] != nxt:
                continue

            # Rule 3: Required predecessors.
            if v in req_predecessors and req_predecessors[v] != prev:
                continue
            if nxt in req_predecessors and nxt != self.depot and req_predecessors[nxt] != v:
                continue

            detour = self.cost_matrix[prev, v] + self.cost_matrix[v, nxt] - self.cost_matrix[prev, nxt]

            revenue_component = node_prizes.get(v, 0.0) if node_prizes is not None else v_waste * self.R

            marginal = revenue_component - detour * self.C - dual_values.get(v, 0.0)

            if marginal > best_marginal:
                best_marginal = marginal
                best_pos = pos

        return best_pos, best_marginal

    def _greedy_route_construction(
        self,
        start_node: int,
        dual_values: Dict[Any, float],
        forbidden_arcs: Set[Tuple[int, int]],
        req_successors: Dict[int, int],
        req_predecessors: Dict[int, int],
        rf_separate_pairs: Set[Tuple[int, int]],
        rf_together_pairs: Set[Tuple[int, int]],
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[int], float]:
        """
        Build a single route greedily from *start_node*.

        At each step the next insertion is chosen to maximise marginal profit
        (revenue + dual benefit − detour cost).  All active branching
        constraints are checked before a candidate node is accepted.

        Constraint enforcement inside the construction loop:

        1. **Forbidden arc** ``(tail, v)``:
           Reject *v* if the arc from the current route tail to *v* is forbidden.

        2. **Required successor** of tail ``→ w``:
           If the current tail has a required outgoing arc to *w*, only *w*
           is a legal next node (any other *v* is rejected).

        3. **Required predecessor** of *v* ``← x``:
           Reject *v* if it requires a predecessor other than the current tail.

        4. **Ryan-Foster separate** ``(r, s)`` with ``together=False``:
           Reject *v* if adding *v* would place it in the same route as a
           node already present that is in a separation pair with *v*.

        After construction, together-pairs are checked at route-completion
        time.  Any route that contains exactly one member of a together-pair
        is discarded (reduced cost set to −∞ so it is filtered out).

        Args:
            start_node: First customer node to seed the route.
            dual_values: Node-coverage and vehicle-limit duals.
            forbidden_arcs: Set of (u, v) pairs that must not be traversed.
            req_successors: u → w meaning u must be immediately followed by w.
            req_predecessors: v → x meaning v must be immediately preceded by x.
            rf_separate_pairs: Sorted node pairs that must be in different routes.
            rf_together_pairs: Sorted node pairs that must be in the same route.

        Returns:
            ``(route_nodes, reduced_cost)`` where route_nodes excludes the depot.
        """
        # ---- Validate start node itself ---------------------------------
        # The depot (0) is the implicit predecessor of start_node.
        if start_node in req_predecessors and req_predecessors[start_node] != self.depot:
            # start_node requires a non-depot predecessor; cannot start here.
            return [], -float("inf")

        route: List[int] = [start_node]
        current_load = self.wastes.get(start_node, 0.0)
        visited: Set[int] = {start_node}

        # ---- Greedy insertion -------------------------------------------
        while True:
            tail = route[-1]

            # If tail has a required successor that is unvisited, it is the
            # *only* valid next node; skip the full candidate scan.
            if tail in req_successors:
                forced = req_successors[tail]
                if forced not in visited and forced != self.depot:
                    forced_waste = self.wastes.get(forced, 0.0)
                    # Check that forced node's predecessor requirement is met and Ryan-Foster separation.
                    if (
                        current_load + forced_waste <= self.capacity
                        and (forced not in req_predecessors or req_predecessors[forced] == tail)
                        and not self._violates_rf_separation(forced, visited, rf_separate_pairs)
                    ):
                        route.append(forced)
                        visited.add(forced)
                        current_load += forced_waste
                        continue
                # Cannot satisfy required successor → stop extending.
                break

            # General case: score all unvisited candidates.
            best_node: Optional[Tuple[int, int]] = None  # (node, insertion_pos)
            best_profit: float = -float("inf")
            for v in range(1, self.n_nodes + 1):
                if v in visited:
                    continue

                v_waste = self.wastes.get(v, 0.0)
                if current_load + v_waste > self.capacity:
                    continue

                pos, marginal = self._evaluate_candidate_insertion(
                    v,
                    v_waste,
                    route,
                    dual_values,
                    forbidden_arcs,
                    req_successors,
                    req_predecessors,
                    rf_separate_pairs,
                    visited,
                    node_prizes,
                )

                if pos is not None and marginal > best_profit:
                    best_profit = marginal
                    best_node = (v, pos)

            if best_node is None or best_profit <= 0:
                break

            v, pos = best_node
            route.insert(pos, v)
            visited.add(v)
            current_load += self.wastes.get(v, 0.0)

        # ---- Together-pair completion check -----------------------------
        # Discard any route that contains exactly one member of a together-pair.
        for r, s in rf_together_pairs:
            r_in = r in visited
            s_in = s in visited
            if r_in != s_in:
                # Together constraint violated at route completion.
                return route, -float("inf")

        reduced_cost = self._compute_reduced_cost(route, dual_values, capacity_cut_duals, node_prizes)
        return route, reduced_cost

    # ------------------------------------------------------------------
    # Ryan-Foster separation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _violates_rf_separation(
        candidate: int,
        visited: Set[int],
        rf_separate_pairs: Set[Tuple[int, int]],
    ) -> bool:
        """
        Return True if adding *candidate* to the route would violate a
        Ryan-Foster separation constraint.

        A violation occurs when *candidate* is in a ``together=False`` pair
        with any node already present in ``visited``.

        Args:
            candidate: Node being evaluated for insertion.
            visited: Nodes already in the partial route.
            rf_separate_pairs: Sorted (min, max) pairs that must be separated.

        Returns:
            True if insertion would violate a separation constraint.
        """
        for already in visited:
            key = (min(candidate, already), max(candidate, already))
            if key in rf_separate_pairs:
                return True
        return False

    # ------------------------------------------------------------------
    # Reduced cost and route details
    # ------------------------------------------------------------------

    def _compute_reduced_cost(
        self,
        route: List[int],
        dual_values: Dict[int, float],
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        Compute the reduced cost of a completed route.

        reduced_cost = profit − Σ_i dual_i − dual_vehicle_limit − crossing_penalty
        profit       = (if node_prizes given) Σ_i node_prizes_i else Σ_i waste_i × R
        cost         = total_distance × C

        Args:
            route: Ordered list of customer nodes (depot excluded).
            dual_values: Node-coverage duals and optional vehicle-limit dual.
            capacity_cut_duals: Optional mapping of node-sets to dual values for active capacity cuts.
            node_prizes: Optional fixed prizes per node replacing standard volumetric revenue.

        Returns:
            Scalar reduced cost.
        """
        total_distance = 0.0
        prev = self.depot
        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node
        total_distance += self.cost_matrix[prev, self.depot]

        if node_prizes is not None:
            revenue = sum(node_prizes.get(n, 0.0) for n in route)
        else:
            total_waste = sum(self.wastes.get(n, 0.0) for n in route)
            revenue = total_waste * self.R

        cost = total_distance * self.C
        profit = revenue - cost

        dual_contribution = sum(dual_values.get(n, 0.0) for n in route)
        vehicle_dual = dual_values.get("vehicle_limit", 0.0)  # type: ignore[call-overload]

        crossing_penalty = 0.0
        if capacity_cut_duals:
            full_path = [self.depot] + route + [self.depot]
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                for node_set, dual in capacity_cut_duals.items():
                    if (u in node_set) != (v in node_set):
                        crossing_penalty += dual

        return profit - dual_contribution - vehicle_dual - crossing_penalty

    def compute_route_details(
        self,
        route: List[int],
    ) -> Tuple[float, float, float, Set[int]]:
        """
        Compute cost, revenue, load, and node coverage for a given route.

        Args:
            route: Ordered list of customer nodes (depot excluded).

        Returns:
            Tuple of (cost, revenue, load, node_coverage).
        """
        total_distance = 0.0
        prev = self.depot
        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node
        total_distance += self.cost_matrix[prev, self.depot]

        total_waste = sum(self.wastes.get(n, 0.0) for n in route)
        revenue = total_waste * self.R
        cost = total_distance * self.C

        return cost, revenue, total_waste, set(route)
