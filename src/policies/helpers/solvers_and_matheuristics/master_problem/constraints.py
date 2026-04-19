"""
Constraint addition and management mixin for VRPPMasterProblem.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np

if TYPE_CHECKING:
    from .model import Route
    from .problem_support import MasterProblemSupport

logger = logging.getLogger(__name__)


class VRPPMasterProblemConstraintsMixin:
    """
    Mixin containing methods for adding cutting planes to the master problem.
    This helps reduce the size of the main model.py file.
    """

    def add_edge_clique_cut(
        self: MasterProblemSupport,
        u: int,
        v: int,
        coefficients: Optional[Dict[int, float]] = None,
        rhs: float = 1.0,
    ) -> bool:
        r"""
        Add an Edge Clique cut specifically covering edge (u, v).

        Formulation:
            Σ_{k ∈ C} λ_k + Σ_{k ∈ \bar{C}} α_k λ_k <= |C| - 1

        Where C is a minimal cover of routes using edge (u, v), and α_k are
        lifting coefficients for routes not in the cover.

        Args:
            u, v: Endpoint nodes of the edge.
            coefficients: Mapping from route index to lifting coefficient α_k.
                If None, defaults to 1.0 for all routes using the edge.
            rhs: Right-hand side of the inequality (|C| - 1).

        Returns:
            True if the cut was added, False if it was redundant.
        """
        if self.model is None or not self.lambda_vars:
            return False

        edge_tuple = (min(u, v), max(u, v))
        key = edge_tuple

        if key in self.active_edge_clique_cuts:
            return False

        lhs = gp.LinExpr()
        found_columns = False
        for idx, route in enumerate(self.routes):
            # Check for edge (u, v) in route path
            nodes = [0] + route.nodes + [0]
            contains_edge = False
            for i in range(len(nodes) - 1):
                if tuple(sorted((nodes[i], nodes[i + 1]))) == edge_tuple:
                    contains_edge = True
                    break

            if contains_edge:
                coeff = 1.0
                if coefficients is not None:
                    coeff = coefficients.get(idx, 1.0)
                lhs.add(self.lambda_vars[idx], coeff)
                found_columns = True

        if not found_columns:
            return False

        constr = self.model.addConstr(lhs <= rhs, name=f"Edge_Clique_{edge_tuple[0]}_{edge_tuple[1]}")
        stored_coeffs = coefficients if coefficients is not None else {}
        self.active_edge_clique_cuts[key] = (constr, stored_coeffs)
        self.global_cut_pool.add_cut("edge_clique", edge_tuple)
        self.model.update()
        return True

    def add_subset_row_cut(
        self: MasterProblemSupport,
        node_set: Union[List[int], Set[int], FrozenSet[int]],
    ) -> bool:
        """
        Add a 3-Subset Row Inequality (3-SRI) to the master problem.

        SRI Formulation:
            Σ_{k} ⌊ 1/2 * |S ∩ Route_k| ⌋ * λ_k <= 1

        For any subset S of 3 nodes, the sum of routes visiting at least 2 nodes
        in S cannot exceed 1. This strengthens the relaxation by cutting off
        fractional solutions where three routes visit pairs (node1, node2),
        (node2, node3), and (node1, node3) with value 0.5 each.

        Args:
            node_set: A list of exactly 3 customer nodes.

        Returns:
            True if the cut was successfully added to the model.
        """
        if self.model is None or len(node_set) != 3 or not self.lambda_vars:
            return False

        nodes = sorted(node_set)
        subset_frozenset = frozenset(nodes)
        if subset_frozenset in self.active_sri_cuts:
            return False

        cut_name = f"SRI_{nodes[0]}_{nodes[1]}_{nodes[2]}"

        lhs = gp.LinExpr()
        found_columns = False
        coeff_dict: Dict[str, float] = {}
        for idx, route in enumerate(self.routes):
            # Check node coverage for SRI violation calculation
            count = sum(1 for n in nodes if n in route.node_coverage)
            coeff = count // 2
            if coeff > 0:
                lhs.add(self.lambda_vars[idx], float(coeff))
                content = ",".join(map(str, route.nodes))
                route_h = hashlib.md5(content.encode()).hexdigest()
                coeff_dict[route_h] = float(coeff)
                found_columns = True

        if not found_columns:
            return False

        new_cut = self.model.addConstr(lhs <= 1.0, name=cut_name)
        self.active_sri_cuts[subset_frozenset] = new_cut
        # Archiving to Global Pool (SRIs are globally valid); store vector for orthogonality checks
        self.global_cut_pool.add_cut("sri", (subset_frozenset, coeff_dict))
        self.model.update()
        return True

    def add_capacity_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
        _skip_pool: bool = False,
    ) -> bool:
        """
        Add a Rounded Capacity Cut (RCC) to the master problem.

        The cut enforces that the number of edges crossing the boundary δ(S)
        of the node set S must be at least twice the minimum number of
        vehicles required to serve S.

        Constraint:  Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k  >=  rhs

        Args:
            node_list: Nodes in set S.
            rhs: Right-hand side (2 * ⌈demand(S) / Q⌉).

        Returns:
            True if the cut was newly added, False if it already exists/failed.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_capacity_cuts:
            # Already have this cut
            return False

        # Build column-based representation of the cut
        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"capacity_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        if is_global and not _skip_pool:
            # Archive with the actual RHS so descendant nodes replay the correct
            # strength (2*ceil(demand(S)/Q)), not a hard-coded placeholder.
            self.global_cut_pool.add_cut("rcc", (node_set, rhs))

        if self.model is not None:
            self.model.update()
        return True

    def add_lci_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Dict[int, float],
        node_alphas: Optional[Dict[int, float]] = None,
        arc: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Add a Lifted Cover Inequality (LCI) to the master problem.

        LCI formulation (Barnhart et al. 2000, §4):
            Σ_{k ∈ C} λ_k + Σ_{k ∉ C} α_k λ_k  ≤  |C| - 1

        LCI cuts are stored in `active_lci_cuts` (separate from `active_capacity_cuts`
        which holds >= RCC constraints) so that dual extraction uses the correct
        sign: Pi >= 0 for a <= constraint in a MAX LP.

        The optional ``node_alphas`` mapping (node_id → lifting coefficient α_i)
        enables the pricing subproblem to apply the correct per-node penalty when
        the cut's dual variable γ is nonzero.  Per Barnhart et al. (2000) §4.2:

            c'_lm^k = c_lm^k + π_lm + α_lm^k · γ_lm

        The optional ``arc`` parameter identifies the specific saturated arc (i, j)
        that generated this LCI (used by SaturatedArcLCIEngine).  When set, the
        pricing DP applies the dual **only** when traversing arc (i, j), matching
        the paper's arc-level formula exactly.  For node/capacity LCI (e.g.
        PhysicalCapacityLCIEngine) leave ``arc=None``; the DP then falls back to
        the node-visit approximation.

        Args:
            node_list: Customer nodes defining the cover set S.
            rhs: Right-hand side of the inequality (|C| - 1 or lifting bound K).
            coefficients: Route-index → route-level lifting coefficient (α_k).
            node_alphas: Node-id → per-node lifting coefficient (α_i).
                         If None, defaults to 1.0 for cover nodes, 0.0 for others.
            arc: Optional source arc (i, j) for arc-saturation LCI.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_lci_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            if idx < len(self.lambda_vars):
                lhs += coeff * self.lambda_vars[idx]

        if lhs.size() == 0:
            return False

        name = f"lci_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= rhs, name=name)
        self.active_lci_cuts[node_set] = constr
        # Store per-node alphas for pricing dual integration.
        effective_node_alphas: Dict[int, float] = node_alphas if node_alphas is not None else {}
        self.active_lci_node_alphas[node_set] = effective_node_alphas
        # Store source arc (None for node/capacity LCI, set for arc-saturation LCI).
        self.active_lci_arcs[node_set] = arc
        # Archive with rhs, route coefficients, node_alphas, AND arc for descendant replay.
        self.global_cut_pool.add_cut("lci", (node_set, rhs, coefficients, effective_node_alphas, arc))
        if self.model is not None:
            self.model.update()
        return True

    def add_set_packing_capacity_cut(self: MasterProblemSupport, node_list: List[int], rhs: float) -> bool:
        """
        Add a Rounded Capacity Cut (RCC) to the master problem.

        The cut enforces that the number of edges crossing the boundary δ(S)
        of the node set S must be at least twice the minimum number of
        vehicles required to serve S.

        Constraint:  Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k  >=  rhs

        Args:
            node_list: Nodes in set S.
            rhs: Right-hand side (2 * ⌈demand(S) / Q⌉).

        Returns:
            True if the cut was newly added, False if it already exists/failed.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_rcc_cuts:
            # Already have this cut
            return False

        # Build column-based representation of the cut
        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"rcc_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_rcc_cuts[node_set] = constr
        self.model.update()
        return True

    def add_sec_cut(
        self: MasterProblemSupport,
        node_list: Union[List[int], Set[int], FrozenSet[int]],
        rhs: float,
        cut_name: str = "",
        global_cut: bool = True,
        node_i: int = -1,
        node_j: int = -1,
        facet_form: str = "2.1",
    ) -> bool:
        """
        Add a Subtour Elimination Cut (SEC) or PC-SEC to the master problem.

        Args:
            node_list: Set of nodes in the subtour.
            rhs: Right-hand side value.
            cut_name: Optional name for the constraint.
            global_cut: If True, the cut is stored in the global registry.
            node_i: Index of node i for Form 2.2 and 2.3 PC-SECs.
            node_j: Index of node j for Form 2.3 PC-SECs.
            facet_form: Form indicator for PC-SECs (e.g. "2.1", "2.3").
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        registry = self.active_sec_cuts if global_cut else self.active_sec_cuts_local

        if node_set in registry:
            return False

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            # We calculate the coefficient C_k for route k in the cut Σ C_k λ_k >= RHS.
            # Form 2.3: Σ x_e >= 2(y_i + y_j - 1)  =>  Σ x_e - 2y_i - 2y_j >= -2.
            # C_k = crossings(route_k, S) - 2 * (1 if i in route_k) - 2 * (1 if j in route_k).
            val = float(self._count_crossings(route, node_set))

            if node_i > 0 and node_i in route.node_coverage:
                val -= 2.0
            if node_j > 0 and node_j in route.node_coverage:
                val -= 2.0

            if abs(val) > 1e-6:
                lhs += val * self.lambda_vars[idx]

        name = cut_name if cut_name else f"sec_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        registry[node_set] = constr

        # Archiving to Global Pool
        if global_cut and node_i < 0 and node_j < 0:
            # Form 2.1 is always valid Σ x_e >= 2
            self.global_cut_pool.add_cut("sec_2.1", node_set)

        self.model.update()
        return True

    def _count_crossings(self: MasterProblemSupport, route: Route, node_set: FrozenSet[int]) -> int:
        """Count how many times a route crosses the boundary δ(S)."""
        crossings = 0
        path_nodes = [0] + route.nodes + [0]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if (u in node_set) != (v in node_set):
                crossings += 1
        return crossings

    def remove_local_cuts(self: MasterProblemSupport) -> int:
        """
        Delete all node-local cuts from the Gurobi model and clear registries.

        Returns:
            Number of cuts removed.
        """
        if self.model is None:
            return 0
        removed = 0
        for constr in self.active_sec_cuts_local.values():
            try:
                self.model.remove(constr)
                removed += 1
            except gp.GurobiError:
                continue

        self.active_sec_cuts_local.clear()
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.model.update()
        return removed

    def find_and_add_violated_rcc(
        self: MasterProblemSupport,
        route_values: Dict[int, float],
        routes: List[Route],
        max_cuts: int = 5,
    ) -> int:
        """
        Separate and add Rounded Capacity Cuts (RCC) based on the current LP solution.

        This follows Section 7 of Barnhart et al. (1998) and Desrochers et al. (1992)
        using a connectivity heuristic:
        1. Build the fractional flow support graph (arcs with x_uv > 0).
        2. Identify connected components (S) of customer nodes.
        3. For each component S, check if the routing bound is violated:
           Σ_{k} x^k(δ(S)) λ_k  <  2 * ⌈ (Σ_{i∈S} waste_i) / Q ⌉
        """
        if not route_values or not routes:
            return 0

        # Aggregate arc flows from active routes
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, val in route_values.items():
            if val < 1e-6 or idx >= len(routes):
                continue
            path = [0] + routes[idx].nodes + [0]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                arc = (u, v)
                arc_flow[arc] = arc_flow.get(arc, 0.0) + val

        # Find connected components of customer nodes in the support graph (ignoring depot)
        components = self._find_customer_components(arc_flow)

        # Evaluate each component S as a cut candidate
        cuts_added = 0
        for S in components:
            if not S:
                continue
            total_waste = sum(self.wastes.get(i, 0.0) for i in S)
            rhs = 2.0 * np.ceil(total_waste / self.capacity) if self.capacity > 0 else 2.0

            # Calculate current LHS value (sum of flow crossing the boundary delta(S))
            lhs_val = sum(flow for (u, v), flow in arc_flow.items() if (u in S) != (v in S))

            # Add if violated by more than 1e-4
            if lhs_val < rhs - 1e-4 and self.add_set_packing_capacity_cut(list(S), rhs):
                cuts_added += 1
                if cuts_added >= max_cuts:
                    break
        return cuts_added

    def _find_customer_components(self: MasterProblemSupport, arc_flow: Dict[Tuple[int, int], float]) -> List[Set[int]]:
        """Identify connected components of customer nodes in the support graph."""
        adj: Dict[int, Set[int]] = {}
        customer_nodes = set()
        for u, v in arc_flow.keys():
            if u != 0 and v != 0:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
                customer_nodes.add(u)
                customer_nodes.add(v)

        visited = set()
        components: List[Set[int]] = []
        for node in customer_nodes:
            if node not in visited:
                comp = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.add(curr)
                        stack.extend(adj.get(curr, set()) - visited)
                components.append(comp)
        return components
