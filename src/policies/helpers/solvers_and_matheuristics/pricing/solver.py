r"""
RCSPP Pricing Subproblem Solver.

Solves the Resource-Constrained Shortest Path Problem (RCSPP) to identify
profitable columns (routes) for the BPC Master Problem.

Theoretical Deviation Note:
---------------------------
Unlike the pure ODIMCF formulation in Barnhart, Hane, and Vance (2000) which
manages state-space explosion strictly via branching, this implementation intentionally
incorporates Subset-Row Inequalities (SRIs) and accommodates Ryan-Foster branching.
These algorithmic choices fundamentally expand the DP state space by requiring auxiliary
resource dimensions (e.g., tracking SRI parity constraints). To mitigate this explosion
and maintain tractability, we rely heavily on the ng-route relaxation framework,
balancing bounding strength with computational viability for the VRPP.
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, cast

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.labels import Label

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint

logger = logging.getLogger(__name__)

# Type alias for LCI cover items passed to the DP extension step.
# Each tuple: (cover_set, node_alpha_dict, dual_value, source_arc_or_none)
# source_arc_or_none is (i, j) for arc-saturation LCI, None for node/capacity LCI.
_LCICoverItem = Tuple[FrozenSet[int], Dict[int, float], float, Optional[Tuple[int, int]]]


class RCSPPSolver:
    """
    Exact / ng-relaxed solver for the Resource-Constrained Shortest Path Problem.
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Optional[Set[int]] = None,
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
        ng_neighborhoods: Optional[Dict[int, Set[int]]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.node_prizes = node_prizes
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.depot = 0
        self.use_ng_routes = use_ng_routes
        self.ng_neighborhood_size = ng_neighborhood_size

        self.labels_generated: int = 0
        self.labels_dominated: int = 0
        self.labels_infeasible: int = 0
        self.last_max_rc: float = -float("inf")
        self.dual_values: Dict[int, float] = {}
        self.bounds_to: np.ndarray = np.zeros(self.n_nodes + 1)
        self.bounds_from: np.ndarray = np.zeros(self.n_nodes + 1)
        self.fixed_arcs: Set[Tuple[int, int]] = set()

        if ng_neighborhoods is not None:
            self.ng_neighborhoods = ng_neighborhoods
        else:
            self.ng_neighborhoods = self._compute_ng_neighborhoods()

        # Dual for the vehicle-limit convexity constraint (set by solve() before use)
        self.vehicle_dual: float = 0.0

        # Precompute distance-sorted neighbor lists
        self._sorted_neighbors: Dict[int, List[int]] = self._precompute_sorted_neighbors()

    def _precompute_sorted_neighbors(self) -> Dict[int, List[int]]:
        """Precompute distance-sorted neighbor lists for all nodes."""
        result: Dict[int, List[int]] = {}
        for i in range(self.n_nodes + 1):
            dists = [(self.cost_matrix[i, j], j) for j in range(self.n_nodes + 1) if j != i]
            dists.sort()
            result[i] = [j for _, j in dists]
        return result

    def _compute_ng_neighborhoods(self) -> Dict[int, Set[int]]:
        customer_nodes = list(range(1, self.n_nodes + 1))
        k = min(self.ng_neighborhood_size, len(customer_nodes))
        neighborhoods: Dict[int, Set[int]] = {}
        for i in customer_nodes:
            distances = sorted((self.cost_matrix[i, j], j) for j in customer_nodes if j != i)
            closest: Set[int] = {j for _, j in distances[: k - 1]}
            closest.add(i)
            neighborhoods[i] = closest
        neighborhoods[self.depot] = set()
        return neighborhoods

    def save_ng_snapshot(self) -> Dict[int, Set[int]]:
        """Return a deep copy of the current ng-neighborhoods."""
        return {k: set(v) for k, v in self.ng_neighborhoods.items()}

    def restore_ng_snapshot(self, snapshot: Dict[int, Set[int]]) -> None:
        """Restore ng-neighborhoods from a previously saved snapshot."""
        self.ng_neighborhoods = {k: set(v) for k, v in snapshot.items()}

    def enforce_elementarity(self, nodes: List[int]) -> int:
        """Dynamically expand ng-sets to enforce strict elementarity on fractional paths."""
        added_count = 0
        nodes_to_enforce = [n for n in nodes if n != self.depot]
        if not nodes_to_enforce:
            return 0
        for i in range(self.n_nodes + 1):
            if i == self.depot:
                continue
            before = len(self.ng_neighborhoods[i])
            self.ng_neighborhoods[i].update(nodes_to_enforce)
            added_count += len(self.ng_neighborhoods[i]) - before
        return added_count

    def expand_ng_neighborhoods(self, cycles: List[Tuple[int, ...]]) -> int:
        added = 0
        for cycle in cycles:
            cset = set(cycle)
            for i in cycle:
                if i == self.depot:
                    continue
                before = len(self.ng_neighborhoods[i])
                self.ng_neighborhoods[i].update(cset)
                added += len(self.ng_neighborhoods[i]) - before
        return added

    def solve(
        self,
        dual_values: Union[Dict[int, float], Dict[str, Any]],
        max_routes: int = 10,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        capacity_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        sri_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        edge_clique_cut_duals: Optional[Dict[Tuple[int, int], float]] = None,
        forced_nodes: Optional[Set[int]] = None,
        rf_conflicts: Optional[Dict[int, Set[int]]] = None,
        is_farkas: bool = False,
        exact_mode: bool = False,
    ) -> List[Route]:
        # Pre-execution setup
        self._rcc_duals_for_bounds = (
            cast(Dict[str, Any], dual_values).get("rcc_duals", {})
            if isinstance(dual_values, dict)
            else capacity_cut_duals or {}
        )

        # 1. Dual handling
        if isinstance(dual_values, dict) and "node_duals" in dual_values:
            complex_duals = cast(Dict[str, Any], dual_values)
            node_duals = complex_duals.get("node_duals", {})
            rcc_duals = complex_duals.get("rcc_duals", {})
            sri_duals = complex_duals.get("sri_duals", {})
            edge_clique_duals = complex_duals.get("edge_clique_duals", {})
            self.vehicle_dual = complex_duals.get("vehicle_limit", 0.0)
            lci_duals_raw = complex_duals.get("lci_duals", {})
            lci_node_alphas_raw = complex_duals.get("lci_node_alphas", {})
            lci_arcs_raw = complex_duals.get("lci_arcs", {})
        else:
            node_duals = dual_values  # type: ignore
            rcc_duals = capacity_cut_duals or {}
            sri_duals = sri_cut_duals or {}
            edge_clique_duals = edge_clique_cut_duals or {}
            self.vehicle_dual = 0.0
            lci_duals_raw = {}
            lci_node_alphas_raw = {}
            lci_arcs_raw = {}

        # Build LCI cover items for DP extension: list of (cover_set, node_alpha, dual, arc_or_none).
        # Only include cuts with non-negligible dual to avoid wasted iteration in the inner loop.
        lci_cover_items: List[_LCICoverItem] = [
            (cover_set, lci_node_alphas_raw.get(cover_set, {}), dual, lci_arcs_raw.get(cover_set))
            for cover_set, dual in lci_duals_raw.items()
            if dual > 1e-8
        ]

        # 2. Reset state
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0
        self.last_max_rc = -float("inf")
        self.dual_values = node_duals
        self.is_farkas = is_farkas
        self.forced_nodes = forced_nodes or set()
        self.rf_conflicts = rf_conflicts or {}

        # Task 2: Completion Bounds
        self._compute_completion_bounds()

        # 3. SRI pre-processing
        active_sri_items = [(k, v) for k, v in sri_duals.items() if v > 1e-4]
        active_sri_items.sort(key=lambda x: x[1], reverse=True)
        active_sri_subsets = sorted([k for k, v in active_sri_items], key=hash)
        sri_dual_values = [sri_duals[s] for s in active_sri_subsets]
        self.sri_dual_values = sri_dual_values
        node_to_sri: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes + 1)}
        for idx, s in enumerate(active_sri_subsets):
            for node in s:
                if node in node_to_sri:
                    node_to_sri[node].append(idx)

        # 4. Precompute reduced costs
        if not self.is_farkas:
            self.node_reduced_costs = {}
            for n in range(1, self.n_nodes + 1):
                rev = self.node_prizes.get(n, 0.0) if self.node_prizes is not None else self.wastes.get(n, 0.0) * self.R
                self.node_reduced_costs[n] = rev - node_duals.get(n, 0.0)
        else:
            self.node_reduced_costs = {n: node_duals.get(n, 0.0) for n in range(1, self.n_nodes + 1)}

        # 5. Constraints
        constraints: List[Any] = branching_constraints or []
        (forbidden_arcs, req_succ, req_pred, rf_sep, rf_tog) = self._preprocess_constraints(constraints)

        original_use_ng = self.use_ng_routes
        try:
            routes = self._label_correcting_algorithm(
                max_routes=max_routes,
                forbidden_arcs=forbidden_arcs,
                required_successors=req_succ,
                required_predecessors=req_pred,
                rf_separate=rf_sep,
                rf_together=rf_tog,
                rcc_duals=rcc_duals,
                active_sri_subsets=active_sri_subsets,
                sri_dual_values=sri_dual_values,
                node_to_sri=node_to_sri,
                edge_clique_duals=edge_clique_duals,
                lci_cover_items=lci_cover_items,
                exact_mode=exact_mode,
            )
        finally:
            self.use_ng_routes = original_use_ng

        routes.sort(key=lambda x: x.reduced_cost, reverse=True)  # type: ignore[arg-type,return-value]
        return routes[:max_routes]

    def _compute_completion_bounds(self):  # noqa: C901
        """
        Compute backward bounds used to aggressively prune DP states.
        """
        # Incorporate RCC duals into completion bounds for tighter pruning.
        # RCC duals are arc-crossing penalties with a clear per-edge interpretation.
        self.bounds_to = np.zeros(self.n_nodes + 1)
        self.bounds_from = np.zeros(self.n_nodes + 1)
        rcc_duals = getattr(self, "_rcc_duals_for_bounds", {})
        nodes = list(range(self.n_nodes + 1))
        for _ in range(self.n_nodes):
            changed = False
            for i in nodes:
                # To Depot
                for j in nodes:
                    if i == j:
                        continue

                    edge_cost = 0.0 if self.is_farkas else (self.cost_matrix[i, j] * self.C)
                    if self.is_farkas:
                        node_rev = 0.0
                    elif self.node_prizes is not None:
                        node_rev = self.node_prizes.get(j, 0.0)
                    else:
                        node_rev = self.wastes.get(j, 0.0) * self.R
                    node_dual = self.dual_values.get(j, 0.0)

                    # Add RCC dual contributions: for each cut set S containing j,
                    # subtract the dual if crossing the cut boundary.
                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if j in S and i not in S)

                    if self.is_farkas:
                        val = (node_dual + rcc_penalty) + self.bounds_to[j]
                    else:
                        val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_to[j]

                    if val > self.bounds_to[i]:
                        self.bounds_to[i] = val
                        changed = True

                # From Depot
                for j in nodes:
                    if i == j:
                        continue

                    edge_cost = 0.0 if self.is_farkas else (self.cost_matrix[j, i] * self.C)
                    if self.is_farkas:
                        node_rev = 0.0
                    elif self.node_prizes is not None:
                        node_rev = self.node_prizes.get(i, 0.0)
                    else:
                        node_rev = self.wastes.get(i, 0.0) * self.R
                    node_dual = self.dual_values.get(i, 0.0)

                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if i in S and j not in S)

                    if self.is_farkas:
                        val = (node_dual + rcc_penalty) + self.bounds_from[j]
                    else:
                        val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_from[j]

                    if val > self.bounds_from[i]:
                        self.bounds_from[i] = val
                        changed = True
            if not changed:
                break

    def _preprocess_constraints(self, constraints: List[Any]):
        forbidden: Set[Tuple[int, int]] = set()
        req_succ: Dict[int, int] = {}
        req_pred: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()
        for c in constraints:
            if hasattr(c, "must_use"):
                if not c.must_use:
                    forbidden.add((c.u, c.v))
                else:
                    req_succ[c.u] = c.v
                    req_pred[c.v] = c.u
            elif hasattr(c, "together"):
                pair = tuple(sorted((c.node_r, c.node_s)))
                if not c.together:
                    rf_separate.add(pair)
                else:
                    rf_together.add(pair)
        return frozenset(forbidden), req_succ, req_pred, rf_separate, rf_together

    def _label_correcting_algorithm(  # noqa: C901
        self,
        max_routes: int,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        required_successors: Dict[int, int],
        required_predecessors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri_subsets: List[FrozenSet[int]],
        sri_dual_values: List[float],
        node_to_sri: Dict[int, List[int]],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
        exact_mode: bool = False,
    ) -> List[Route]:
        """Forward label-correcting algorithm (priority-queue order)."""
        use_ng = self.use_ng_routes
        initial_sri = tuple([0] * len(active_sri_subsets))

        start = Label(reduced_cost=0.0, node=self.depot, load=0.0, sri_state=initial_sri)

        # Max-heap via negated reduced_cost; integer counter breaks ties.
        queue: List[Tuple[float, int, Label]] = [(-start.reduced_cost, 0, start)]
        _counter = 0

        labels_at_node: Dict[int, List[Label]] = {self.depot: [start]}
        completed_routes: List[Label] = []
        global_max_rc = -float("inf")

        while queue:
            _, _, current = heapq.heappop(queue)
            u = current.node

            # Candidate successors
            neighbor_limit = self.n_nodes if exact_mode else 20
            candidates = (
                [required_successors[u]] if u in required_successors else self._get_neighbors(u, neighbor_limit)
            )

            for v in candidates:
                # Fix 2: Early exit for fixed arcs before expensive feasibility checks
                if (u, v) in self.fixed_arcs:
                    continue

                # Elementarity
                if v in current.visited:
                    continue

                # ng-feasibility
                if (use_ng and v in current.ng_memory) and u not in required_successors:
                    continue

                # Ryan-Foster separation
                if any(tuple(sorted((v, n))) in rf_separate for n in current.visited):
                    continue

                # Edge constraints
                if (u, v) in forbidden_arcs:
                    continue
                if v in required_predecessors and required_predecessors[v] != u:
                    continue

                new_label = self._extend_label(
                    current,
                    v,
                    forbidden_arcs,
                    rcc_duals,
                    active_sri_subsets,
                    sri_dual_values,
                    node_to_sri,
                    edge_clique_duals,
                    lci_cover_items,
                )
                if new_label is None:
                    continue

                self.labels_generated += 1
                existing = labels_at_node.get(v, [])
                if any(e.dominates(new_label, use_ng=use_ng, sri_dual_values=sri_dual_values) for e in existing):
                    self.labels_dominated += 1
                    continue

                labels_at_node[v] = [lbl for lbl in existing if not new_label.dominates(lbl, use_ng=use_ng)]
                labels_at_node[v].append(new_label)
                _counter += 1
                heapq.heappush(queue, (-new_label.reduced_cost, _counter, new_label))
                global_max_rc = max(global_max_rc, new_label.reduced_cost)

            # Attempt depot return
            if u != self.depot:
                can_return = not (u in required_successors and required_successors[u] != self.depot)

                # Completion bounds pruning.
                # Only skip the depot-return attempt when the BEST possible final
                # reduced cost from this label is provably <= 0 — i.e., even the
                # optimal completion cannot produce a column worth adding.
                if current.reduced_cost + self.bounds_to[u] > 0.0 and can_return:
                    final = self._extend_to_depot(current)  # type: ignore[arg-type]
                    if final:
                        completed_routes.append(final)
                        global_max_rc = max(global_max_rc, final.reduced_cost)

        self.last_max_rc = global_max_rc
        routes: List[Route] = []
        for label in sorted(completed_routes, key=lambda x: x.reduced_cost, reverse=True):
            if label.reduced_cost > 1e-6:
                path = label.reconstruct_path()
                nodes = [n for n in path if n != self.depot]
                rt = self._compute_route_details(nodes)
                rt.reduced_cost = label.reduced_cost
                routes.append(rt)
                if len(routes) >= max_routes:
                    break
        return routes

    def _get_neighbors(self, node: int, limit: int) -> List[int]:
        """
        Return the nearest neighbors for a given node.

        Used to artificially bound the RCSPP transition state-space during
        heuristic pricing passes by limiting expansions to the closest
        candidates (e.g., limit=20).
        """
        return self._sorted_neighbors[node][:limit]

    def _extend_to_depot(self, label: Label) -> Optional[Label]:
        edge_cost = (self.cost_matrix[label.node, self.depot] * self.C) if not self.is_farkas else 0.0

        # RCC crossing penalty for the depot-return arc.
        # The depot (node 0) is never inside a customer cut set S, so an arc
        # from label.node ∈ S back to the depot is always a boundary exit crossing
        # and must carry the RCC dual penalty.
        crossing_penalty = sum(
            mu
            for S, mu in getattr(self, "_rcc_duals_for_bounds", {}).items()
            if label.node in S and self.depot not in S
        )

        if self.is_farkas:
            # Add duals to maximize PI * A
            new_rc = label.reduced_cost - edge_cost + self.vehicle_dual + crossing_penalty
        else:
            new_rc = label.reduced_cost - edge_cost - self.vehicle_dual - crossing_penalty

        return Label(
            reduced_cost=new_rc,
            node=self.depot,
            load=label.load,
            visited=label.visited,
            ng_memory=label.ng_memory,
            rf_unmatched=label.rf_unmatched,
            parent=label,
            sri_state=label.sri_state,
        )

    def _compute_route_details(self, nodes: List[int]) -> Route:
        """Compute cost/revenue/load for a customer-only node list (no depot bookends)."""
        cost = 0.0
        prev = self.depot
        for n in nodes:
            cost += self.cost_matrix[prev, n] * self.C
            prev = n
        cost += self.cost_matrix[prev, self.depot] * self.C

        if self.node_prizes is not None:
            revenue = sum(self.node_prizes.get(n, 0.0) for n in nodes)
        else:
            revenue = sum(self.wastes.get(n, 0.0) for n in nodes) * self.R

        waste = sum(self.wastes.get(n, 0.0) for n in nodes)
        return Route(nodes, cost, revenue, waste, set(nodes))

    def _extend_label(
        self,
        label: Label,
        next_node: int,
        forbidden: FrozenSet[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri: List[FrozenSet[int]],
        sri_duals: List[float],
        node_to_sri: Dict[int, List[int]],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
    ) -> Optional[Label]:
        if (label.node, next_node) in forbidden:
            return None

        load = label.load + self.wastes.get(next_node, 0.0)
        if load > self.capacity + 1e-6:
            return None

        # Task 1: Basic BC calculation (profit - dist - duals)
        dist = self.cost_matrix[label.node, next_node]
        cost = (dist * self.C) if not self.is_farkas else 0.0

        if self.node_prizes is not None:
            rev = self.node_prizes.get(next_node, 0.0) if not self.is_farkas else 0.0
        else:
            rev = (self.wastes.get(next_node, 0.0) * self.R) if not self.is_farkas else 0.0

        rc_delta = (
            (rev - cost - self.dual_values.get(next_node, 0.0))
            if not self.is_farkas
            else self.dual_values.get(next_node, 0.0)
        )

        # RCC duals: penalise each boundary CROSSING, not each node visit.
        # A crossing occurs when the arc (label.node → next_node) transitions
        # between the interior and exterior of cut set S.
        for subset, dual in rcc_duals.items():
            if (label.node in subset) != (next_node in subset):
                rc_delta += dual if self.is_farkas else -dual

        # Edge-Clique dual penalty (Barnhart et al. 2000, §4.2).
        # For a cut on edge (u, v) with dual γ, every route traversing (u, v)
        # must have its reduced cost decreased by γ.
        if edge_clique_duals:
            can_edge = tuple(sorted((label.node, next_node)))
            rc_delta -= edge_clique_duals.get(can_edge, 0.0)  # type: ignore[arg-type]

        # Fix 4: SRI Dual Penalty
        # 1st visit: no penalty yet.
        # 2nd visit: apply penalty exactly once.
        new_sri = list(label.sri_state)
        for idx in node_to_sri.get(next_node, []):
            if new_sri[idx] == 0:
                new_sri[idx] = 1
            elif new_sri[idx] == 1:
                new_sri[idx] = 2
                rc_delta -= sri_duals[idx]

        # LCI Dual Penalty — Barnhart, Hane, Vance (2000) §4.2
        # Two penalty modes depending on cut origin:
        # 1. Arc-saturation LCI
        # 2. Node/capacity LCI
        for cover_set, node_alpha, dual, lci_arc in lci_cover_items:
            if lci_arc:
                if (label.node, next_node) == lci_arc:
                    rc_delta -= dual
            elif next_node in cover_set:
                rc_delta -= node_alpha.get(next_node, 1.0) * dual

        # NG-Memory transition
        return Label(
            node=next_node,
            load=load,
            reduced_cost=label.reduced_cost + rc_delta,
            visited=label.visited | {next_node},
            parent=label,
            ng_memory=(label.ng_memory & self.ng_neighborhoods[next_node]) | {next_node},
            sri_state=tuple(new_sri),
            rf_unmatched=label.rf_unmatched,
        )
