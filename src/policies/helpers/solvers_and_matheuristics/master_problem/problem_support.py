"""
Constraint addition and management mixin for VRPPMasterProblem.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
    from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool import GlobalCutPool

logger = logging.getLogger(__name__)


class MasterProblemSupport(Protocol):
    """
    Structural interface defining the requirements for a VRPP Master Problem.
    """

    n_nodes: int
    mandatory_nodes: Set[int]
    optional_nodes: Set[int]
    cost_matrix: np.ndarray[Any, Any]
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    vehicle_limit: Optional[int]
    global_cut_pool: "GlobalCutPool"
    BIG_M: float

    model: gp.Model
    routes: List[Route]
    lambda_vars: List[gp.Var]
    dual_node_coverage: Dict[int, float]
    dual_vehicle_limit: float
    global_column_pool: List[Route]
    phase: int

    # Active cuts registries
    active_src_cuts: Dict[FrozenSet[int], gp.Constr]
    active_sec_cuts: Dict[FrozenSet[int], gp.Constr]
    active_sec_cuts_local: Dict[FrozenSet[int], gp.Constr]
    active_rcc_cuts: Dict[FrozenSet[int], gp.Constr]
    active_capacity_cuts: Dict[FrozenSet[int], gp.Constr]
    active_lci_cuts: Dict[FrozenSet[int], gp.Constr]
    active_lci_node_alphas: Dict[FrozenSet[int], Dict[int, float]]
    active_lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]]
    active_sri_cuts: Dict[FrozenSet[int], gp.Constr]
    active_edge_clique_cuts: Dict[Tuple[int, int], Tuple[gp.Constr, Dict[int, float]]]

    # Dual value registries
    dual_src_cuts: Dict[FrozenSet[int], float]
    dual_sec_cuts: Dict[FrozenSet[int], float]
    dual_sec_cuts_local: Dict[FrozenSet[int], float]
    dual_rcc_cuts: Dict[FrozenSet[int], float]
    dual_capacity_cuts: Dict[FrozenSet[int], float]
    dual_lci_cuts: Dict[FrozenSet[int], float]
    dual_sri_cuts: Dict[FrozenSet[int], float]
    dual_edge_clique_cuts: Dict[Tuple[int, int], float]

    # Dual stabilization configurations
    dual_smoothing_alpha: float
    prev_dual_node_coverage: Dict[int, float]
    prev_dual_vehicle_limit: float
    prev_dual_capacity_cuts: Dict[FrozenSet[int], float]
    prev_dual_sri_cuts: Dict[FrozenSet[int], float]
    farkas_duals: Dict[str, Dict[Any, float]]

    column_deletion_enabled: bool
    strict_set_partitioning: bool
    enable_dual_smoothing: bool

    # Core Methods
    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None: ...

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]: ...

    def _handle_infeasibility(self) -> Tuple[float, Dict[int, float]]: ...

    def _extract_duals(self) -> None: ...

    def _apply_dual_smoothing(self) -> None: ...

    def solve_ip(self) -> Tuple[float, List[Route]]: ...

    def add_route(self, route: Route) -> None: ...

    def _add_column_to_model(self, route: Route) -> None: ...

    def _wire_route_into_active_cuts(self, route: Route, var: Any) -> None: ...

    def purge_useless_columns(self, tolerance: float = -0.1) -> int: ...

    def get_reduced_cost_coefficients(self) -> Dict[str, Any]: ...

    def get_node_visitation(self) -> Dict[int, float]: ...

    def get_edge_usage(self, only_elementary: bool = False) -> Dict[Tuple[int, int], float]: ...

    def save_basis(self) -> Optional[Tuple[List[int], List[int]]]: ...

    def restore_basis(self, vbasis: List[int], cbasis: List[int]) -> None: ...

    # Problem-support methods
    def sift_global_column_pool(
        self: MasterProblemSupport,
        node_duals: Dict[int, float],
        rcc_duals: Dict[FrozenSet[int], float],
        sri_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_duals: Optional[Dict[FrozenSet[int], float]] = None,
        lci_node_alphas: Optional[Dict[FrozenSet[int], Dict[int, float]]] = None,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        rc_tolerance: float = 1e-5,
    ) -> int: ...

    def calculate_reduced_cost(self: MasterProblemSupport, route: Route, dual_values: Dict[str, Any]) -> float: ...

    def set_phase(self: MasterProblemSupport, phase: int) -> None: ...

    def deduplicate_column_pool(self: MasterProblemSupport, tol: float = 1e-6) -> int: ...

    def has_artificial_variables_active(self: MasterProblemSupport, tol: float = 1e-6) -> bool: ...

    # Constraint (cut) management methods
    def add_edge_clique_cut(
        self: MasterProblemSupport,
        u: int,
        v: int,
        coefficients: Optional[Dict[int, float]] = None,
        rhs: float = 1.0,
    ) -> bool: ...

    def add_subset_row_cut(
        self: MasterProblemSupport,
        node_set: Union[List[int], Set[int], FrozenSet[int]],
    ) -> bool: ...

    def add_capacity_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
        _skip_pool: bool = False,
    ) -> bool: ...

    def add_lci_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Dict[int, float],
        node_alphas: Optional[Dict[int, float]] = None,
        arc: Optional[Tuple[int, int]] = None,
    ) -> bool: ...

    def add_set_packing_capacity_cut(self: MasterProblemSupport, node_list: List[int], rhs: float) -> bool: ...

    def add_sec_cut(
        self: MasterProblemSupport,
        node_list: Union[List[int], Set[int], FrozenSet[int]],
        rhs: float,
        cut_name: str = "",
        global_cut: bool = True,
        node_i: int = -1,
        node_j: int = -1,
        facet_form: str = "2.1",
    ) -> bool: ...

    def _count_crossings(self: MasterProblemSupport, route: Route, node_set: FrozenSet[int]) -> int: ...

    def remove_local_cuts(self: MasterProblemSupport) -> int: ...

    def find_and_add_violated_rcc(
        self,
        route_values: Dict[int, float],
        routes: List[Route],
        max_cuts: int = 5,
    ) -> int: ...

    def _find_customer_components(self, arc_flow: Dict[Tuple[int, int], float]) -> List[Set[int]]: ...


class VRPPMasterProblemSupportMixin:
    """
    Mixin containing methods that support the master problem.
    This helps reduce the size of the main model.py file.
    """

    def sift_global_column_pool(
        self: MasterProblemSupport,
        node_duals: Dict[int, float],
        rcc_duals: Dict[FrozenSet[int], float],
        sri_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_duals: Optional[Dict[FrozenSet[int], float]] = None,
        lci_node_alphas: Optional[Dict[FrozenSet[int], Dict[int, float]]] = None,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        rc_tolerance: float = 1e-5,
    ) -> int:
        """
        Scan the global pool for profitable columns under current duals and branching.

        Args:
            node_duals: Base duals from coverage and vehicle limits.
            rcc_duals: Current Root-Capacity Cut duals.
            sri_duals: Current Subset-Row Inequality duals.
            edge_clique_duals: Current Edge Clique cut duals.
            lci_duals: Current Lifted Cover Inequality duals (γ per cover set S).
            lci_node_alphas: Per-node lifting coefficients for LCI pricing adjustment.
            branching_constraints: List of active B&B branching constraints.
            rc_tolerance: Numerical threshold to prevent injection of mathematically
                         insignificant columns (epsilon deadlock).

        Returns:
            Number of routes re-activated and added to the RMP.
        """
        if not self.global_column_pool:
            return 0

        # Bundle duals for calculate_reduced_cost
        dual_values: Dict[str, Any] = {
            "node_duals": node_duals,
            "rcc_duals": rcc_duals,
            "sri_duals": sri_duals,
            "edge_clique_duals": edge_clique_duals,
            "lci_duals": lci_duals or {},
            "lci_node_alphas": lci_node_alphas or {},
        }

        # We use a set of signatures to prevent adding duplicates if they already exist in RMP
        # (Though column generation usually handles this via dominance).
        active_sigs = {tuple(r.nodes) for r in self.routes}

        # Task 6 (SOTA): Elite Sifting.
        # Instead of adding all profitable columns, we sort by current reduced cost
        # and only inject the top 'max_to_add' (N=30) movers. This prevents RMP bloat.
        max_to_add = 30
        candidates: List[Tuple[float, int]] = []

        for i, route in enumerate(self.global_column_pool):
            # Task 1: Enforce physical elementarity.
            if len(set(route.nodes)) != len(route.nodes):
                continue

            # Check for duplicates in active model
            if tuple(route.nodes) in active_sigs:
                continue

            # Task 6: Check feasibility against branching constraints.
            if branching_constraints and not all(bc.is_route_feasible(route) for bc in branching_constraints):
                continue

            # Task 1: Enforce rc_tolerance
            rc = self.calculate_reduced_cost(route, dual_values)
            if rc > rc_tolerance:
                candidates.append((rc, i))

        if not candidates:
            return 0

        # SORT BY QUALITY (Elite Sifting)
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_elite = candidates[:max_to_add]

        # Extract only indices for removal (reverse to preserve indexing)
        top_indices = sorted([idx for _, idx in top_elite], reverse=True)

        added = 0
        for idx in top_indices:
            route = self.global_column_pool.pop(idx)
            self.add_route(route)
            added += 1

        if added > 0 and self.model is not None:
            self.model.update()
            logger.info(f"[Sifting] Re-added {added} elite columns from pool (cap={max_to_add}).")
        return added

    def calculate_reduced_cost(self: MasterProblemSupport, route: Route, dual_values: Dict[str, Any]) -> float:
        """
        Helper to calculate reduced cost of a route using current duals.
        Correctly accounts for node coverage, fleet limit, and all active cuts.
        """
        node_duals = dual_values.get("node_duals", {})

        # Reduced cost base: Profit - Σ duals
        # For Set Packing (>= 1 or == 1), duals π_i are represented such that
        # rc = p_k - Σ a_ik * π_i.
        rc = route.revenue - route.cost

        # 1. Node coverage duals (Partitioning/Packing)
        for node in route.node_coverage:
            rc -= node_duals.get(node, 0.0)

        # 2. Vehicle limit dual
        # If dual_values contains the key, it's already extracted from get_reduced_cost_coefficients
        # which bundles it into node_duals["vehicle_limit"].
        rc -= node_duals.get("vehicle_limit", 0.0)

        # 3. Capacity Cut (RCC) duals
        rcc_duals = dual_values.get("rcc_duals", {})
        for node_set, dual in rcc_duals.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                rc -= float(crossings) * dual

        # 4. Subset-Row Inequality (SRI) duals
        sri_duals = dual_values.get("sri_duals", {})
        route_nodes = set(route.nodes)
        for subset, dual in sri_duals.items():
            count = len(subset.intersection(route_nodes))
            coeff = count // 2
            if coeff > 0:
                rc -= float(coeff) * dual

        # 5. Edge Clique duals
        edge_clique_duals = dual_values.get("edge_clique_duals", {})
        nodes = [0] + route.nodes + [0]
        for i in range(len(nodes) - 1):
            edge = (min(nodes[i], nodes[i + 1]), max(nodes[i], nodes[i + 1]))
            if edge in edge_clique_duals:
                rc -= edge_clique_duals[edge]

        # 6. Lifted Cover Inequality (LCI) duals — Barnhart et al. (2000) §4.2
        # For each active LCI with cover set S and dual γ_S, the reduced cost
        # contribution is:  rc -= Σ_{i ∈ route ∩ S} α_i · γ_S
        # where α_i is the node-level lifting coefficient.
        lci_duals = dual_values.get("lci_duals", {})
        lci_node_alphas = dual_values.get("lci_node_alphas", {})
        if lci_duals:
            route_nodes_set = route.node_coverage
            for cover_set, dual in lci_duals.items():
                if dual < 1e-9:
                    continue
                node_alpha = lci_node_alphas.get(cover_set, {})
                for node in route_nodes_set:
                    alpha = node_alpha.get(node, 1.0 if node in cover_set else 0.0)
                    if alpha > 1e-9:
                        rc -= alpha * dual

        return rc

    def set_phase(self: MasterProblemSupport, phase: int) -> None:
        """
        Switch between Phase 1 (Feasibility) and Phase 2 (Optimality).

        Phase 1: All route objectives are set to 0.0 so the LP has no inherent
            bias.  Gurobi will find the RMP infeasible (no Big-M artificials),
            and the Farkas dual ray guides pricing toward columns that restore
            coverage of mandatory nodes.
        Phase 2: Route objectives are restored to their profit values so the LP
            maximises total profit (standard column generation).

        Args:
            phase: 1 or 2.
        """
        if self.model is None:
            return

        self.phase = phase
        if phase == 1:
            logger.info("Master Problem: Switching to Phase I (Feasibility).")
            # Phase I: all route variables have zero obj. coefficient.
            # Farkas dual extraction will guide pricing to find a feasible basis.
            for var in self.lambda_vars:
                var.Obj = 0.0
            self.model.ModelSense = GRB.MAXIMIZE
        else:
            logger.info("Master Problem: Switching to Phase II (Optimality).")
            # Phase II: standard profit maximisation.
            for idx, route in enumerate(self.routes):
                self.lambda_vars[idx].Obj = route.profit
            self.model.ModelSense = GRB.MAXIMIZE

        self.model.update()

    def deduplicate_column_pool(self: MasterProblemSupport, tol: float = 1e-6) -> int:
        """
        Prune mathematically equivalent routes from the global column pool.
        Uses MD5 content hashes of node sequences for O(1) deduplication.
        """
        if not self.global_column_pool:
            return 0

        initial_count = len(self.global_column_pool)
        seen_hashes: Dict[str, float] = {}
        unique_pool = []

        for route in self.global_column_pool:
            content = ",".join(map(str, route.nodes))
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes[h] = route.profit
                unique_pool.append(route)
            else:
                if route.profit > seen_hashes[h] + tol:
                    # Replace existing with higher profit version
                    for idx, r in enumerate(unique_pool):
                        r_content = ",".join(map(str, r.nodes))
                        rh = hashlib.md5(r_content.encode()).hexdigest()
                        if rh == h:
                            unique_pool[idx] = route
                            seen_hashes[h] = route.profit
                            break

        self.global_column_pool = unique_pool
        return initial_count - len(unique_pool)

    def has_artificial_variables_active(self: MasterProblemSupport, tol: float = 1e-6) -> bool:
        """
        Check whether any artificial variable is non-zero in the current solution.

        A non-zero α_i after solving the IP indicates that node i cannot be
        covered by the current column pool under the active branching constraints,
        so the B&B node should be declared infeasible and pruned.

        Args:
            tol: Threshold below which a variable value is considered zero.

        Returns:
            True if at least one α_i > tol, indicating infeasibility.
        """
        # artificial_vars have been removed; infeasibility is now signalled
        # by GRB.INFEASIBLE status directly.
        if self.model is None:
            return False
        return self.model.Status == GRB.INFEASIBLE
