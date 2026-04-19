"""
Master Problem Model for VRPP Branch-and-Price-and-Cut.

Implements the Set Partitioning Problem (SPP) formulation where each column
represents a feasible route. The master problem selects routes to maximize
total collected profit across a finite fleet.

Theoretical Basis: Barnhart et al. (1998).
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.constraints import (
    VRPPMasterProblemConstraintsMixin,
)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool import GlobalCutPool
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support import (
    MasterProblemSupport,
    VRPPMasterProblemSupportMixin,
)

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

logger = logging.getLogger(__name__)


class VRPPMasterProblem(VRPPMasterProblemConstraintsMixin, VRPPMasterProblemSupportMixin, MasterProblemSupport):
    """
    Master Problem (MP) for the Branch-and-Price-and-Cut algorithm.

    Formulates and solves the Set Partitioning Problem (SPP) relaxation of the VRPP.
    The Master Problem is the core coordinator for the column generation loop,
    managing the interaction between the linear relaxation (Gurobi), the
    cutting plane engines, and the branching constraints.

    Mathematical Formulation:
    -------------------------
    Maximize ∑_{k ∈ Ω} p_k * λ_k
    Subject to:
        ∑_{k ∈ Ω} a_{ik} * λ_k = 1           ∀ i ∈ Mandatory Customers (Set Partitioning)
        ∑_{k ∈ Ω} a_{jk} * λ_k ≤ 1           ∀ j ∈ Optional Customers (Set Packing)
        ∑_{k ∈ Ω} λ_k <= K                   (Vehicle Fleet Constraint / Knapsack)
        ∑_{k ∈ Ω} γ_{Sk} * λ_k ≤ RHS_S       (Cutting Planes: RCC, SRI, Fleet Cover)
        λ_k ∈ {0, 1}

    Theoretical Exactness & Farkas Pricing:
    ---------------------------------------
    To maintain theoretical exactness and support Ryan-Foster branching, this
    implementation utilizes strict Set Partitioning for all mandatory nodes.
    In the event of LP infeasibility (common during the early stages of
    branching), the RMP utilizes a 2-Phase approach. Instead of using
    arbitrary Big-M artificial variables which can cause numerical instability,
    we extract the `FarkasDual` ray from Gurobi. This dual ray points in the
    direction of infeasibility, allowing the pricing subproblem to generate
    columns that specifically resolve the unmet coverage requirements.

    Dual Smoothing & Stabilization:
    -------------------------------
    To mitigate the "tailing-off" effect and dual degeneracy in column
    generation, this MP supports Exponential Dual Smoothing (EDS) as
    proposed by Wentges (1997). This stabilizes the dual values π passed
    to the subproblem, accelerating convergence by preventing the
    dual signal from oscillating between extreme basin points.

    Reference:
        - Barnhart, C., et al. (1998, 2000).
        - Wentges, T. (1997). "Weighted Dantzig-Wolfe decomposition."
    """

    def __init__(
        self,
        n_nodes: int,
        mandatory_nodes: Set[int],
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        vehicle_limit: Optional[int] = None,
        global_cut_pool: Optional[GlobalCutPool] = None,
    ) -> None:
        """
        Initialise the Master Problem for VRPP.

        Framework (Barnhart, Hane, and Vance 2000):
        This class maintains the Restricted Master Problem (RMP) using the
        Barnhart-style sequencing of CG followed by separation.

        Set Partitioning & Branching:
        To maintain Ryan-Foster (1981) compatibility, we use strict equality (== 1)
        for mandatory nodes. Artificial variables (Big-M) are omitted in favor of
        Farkas dual extraction (Phase I), ensuring that primary infeasibility guides
        the search for feasible basis columns.

        Dual Stabilization:
        Includes support for Exponential Dual Smoothing (Wentges, 1997) to
        stabilize the price signal and mitigate the "heading-in" effect.

        Args:
            n_nodes: Number of customer nodes (excluding depot).
            mandatory_nodes: Set of node indices that must be visited.
            cost_matrix: Distance matrix where [0] is the depot.
            wastes: Waste volume (kg) per node.
            capacity: Vehicle payload capacity (kg).
            revenue_per_kg: Revenue coefficient (R).
            cost_per_km: Operating cost coefficient (C).
            vehicle_limit: Maximum fleet size (K).
            global_cut_pool: Central repository for cross-node valid inequalities.
        """
        self.n_nodes = n_nodes
        self.mandatory_nodes = mandatory_nodes
        self.optional_nodes = set(range(1, n_nodes + 1)) - mandatory_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.vehicle_limit = vehicle_limit
        self.global_cut_pool = global_cut_pool or GlobalCutPool()

        # BIG_M Calculation for potential fallbacks
        max_single_node_revenue = max(self.wastes.values(), default=0.0) * self.R
        min_demand = max(1.0, min((w for w in self.wastes.values() if w > 0), default=1.0))
        max_nodes_per_route = max(1, int(self.capacity / min_demand))
        max_route_profit = max_single_node_revenue * min(max_nodes_per_route, self.n_nodes)
        self.BIG_M = max(1000.0, 10.0 * max_route_profit)

        self.model: gp.Model
        self.routes: List[Route] = []
        self.lambda_vars: List[gp.Var] = []

        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_vehicle_limit: float = 0.0
        self.global_column_pool: List[Route] = []
        self.phase: int = 2

        # Active cuts registries
        self.active_src_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts_local: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_rcc_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_capacity_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_lci_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_lci_node_alphas: Dict[FrozenSet[int], Dict[int, float]] = {}
        self.active_lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}
        self.active_sri_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_edge_clique_cuts: Dict[Tuple[int, int], Tuple[gp.Constr, Dict[int, float]]] = {}

        # Dual value registries
        self.dual_src_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.dual_rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_lci_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sri_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_edge_clique_cuts: Dict[Tuple[int, int], float] = {}

        # Dual stabilization configurations
        self.dual_smoothing_alpha: float = 0.5
        self.prev_dual_node_coverage: Dict[int, float] = {}
        self.prev_dual_vehicle_limit: float = 0.0
        self.prev_dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
        self.prev_dual_sri_cuts: Dict[FrozenSet[int], float] = {}
        self.farkas_duals: Dict[str, Dict[Any, float]] = {}

        self.column_deletion_enabled: bool = True
        self.strict_set_partitioning: bool = True
        self.enable_dual_smoothing: bool = False

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the set-covering master problem.

        Implements the Restricted Master Problem (RMP) formulation:

        1. Mandatory-node coverage constraints use == 1 (Set Partitioning) to
           ensure that Ryan-Foster branching logic remains mathematically valid.
        2. Artificial variables (Big-M) are intentionally omitted.  When the
           column pool cannot cover a mandatory node, Gurobi returns INFEASIBLE
           and we extract the Farkas dual ray to guide Phase I pricing toward
           feasibility (Barnhart et al. 2000, §3).  Big-M variables would
           suppress infeasibility and deadlock the Phase I / Farkas loop.
        3. Optional-node constraints are packing inequalities (<= 1).

        Args:
            initial_routes: If provided, replaces self.routes with this list
                before building the model.
        """
        self.model = gp.Model("VRPP_Master")
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1  # Dual Simplex is preferred for SPP/CG re-optimizations
        self.model.Params.InfUnbdInfo = 1
        self.model.Params.DualReductions = 0

        self.lambda_vars = []
        if initial_routes is not None:
            self.routes = initial_routes

        for idx, route in enumerate(self.routes):
            var = self.model.addVar(obj=route.profit, vtype=GRB.BINARY, name=f"route_{idx}")
            self.lambda_vars.append(var)

        # Coverage Constraints
        for node in self.mandatory_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]
            if self.strict_set_partitioning:
                self.model.addConstr(lhs == 1.0, name=f"coverage_{node}")
            else:
                self.model.addConstr(lhs >= 1.0, name=f"coverage_{node}")

        for node in self.optional_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]
            self.model.addConstr(lhs <= 1, name=f"coverage_{node}")

        # Fleet limit Constraint
        if self.vehicle_limit is not None:
            self.model.addConstr(gp.quicksum(self.lambda_vars) <= self.vehicle_limit, name="vehicle_limit")

        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]:
        """
        Solve the LP relaxation of the RMP and extract dual values.

        The integrality constraints on all λ variables are temporarily relaxed
        to continuous [0, 1]. The model is solved using the Dual Simplex method
        (Reference: Barnhart et al. 1998) to ensure efficient warm-starting.

        Dual Extraction & Interpretation:
        - Mandatory Nodes (== 1): π_i is the shadow price.
        - Optional Nodes (<= 1): π_j >= 0.
        - Vehicle Limit (<= K): π_K >= 0 (fleet opportunity cost).
        - Cuts (Σ aλ <= RHS): Dual weights used to penalize "violated" routes in pricing.

        Infeasibility & Farkas:
        If the current column pool cannot satisfy mandatory coverage, Gurobi returns
        GRB.INFEASIBLE. We then extract the Farkas Dual Ray (FarkasDual) which
        mathematically identifies the direction of infeasibility, guiding the
        pricing subproblem to find feasible columns (Phase I).

        Returns:
            Tuple of (LP objective value, {route_index: λ_k value}).
            If infeasible, returns (-inf, {}) and populates self.farkas_duals.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Relax integrality
        for var in self.lambda_vars:
            var.VType = GRB.CONTINUOUS

        self.model.update()
        if self.model.NumVars == 0:
            return 0.0, {}

        self.model.optimize()
        status = self.model.Status

        if status == GRB.INFEASIBLE:
            return self._handle_infeasibility()

        if status != GRB.OPTIMAL:
            logger.warning(f"LP solve returned non-optimal status {status}.")
            return 0.0, {}

        obj_value = self.model.ObjVal
        try:
            route_values = {idx: var.X for idx, var in enumerate(self.lambda_vars)}
        except (gp.GurobiError, AttributeError):
            return 0.0, {}

        self._extract_duals()
        return obj_value, route_values

    def _handle_infeasibility(self) -> Tuple[float, Dict[int, float]]:
        """
        Handle infeasibility by extracting the Farkas dual ray.

        When the RMP is infeasible, the Farkas dual ray provides a certificate
        of infeasibility and guides the pricing subproblem toward finding
        feasible columns (Phase I).

        Returns:
            Tuple of (-inf, {}) and populates self.farkas_duals.
        """
        try:
            farkas_node_duals: Dict[Union[int, str], float] = {}
            for node in range(1, self.n_nodes + 1):
                constr = self.model.getConstrByName(f"coverage_{node}")
                if constr is not None:
                    farkas_node_duals[node] = constr.FarkasDual

            if self.vehicle_limit is not None:
                constr = self.model.getConstrByName("vehicle_limit")
                if constr is not None:
                    farkas_node_duals["vehicle_limit"] = constr.FarkasDual

            self.farkas_duals = {
                "node_duals": farkas_node_duals,
                "rcc_duals": {s: c.FarkasDual for s, c in self.active_capacity_cuts.items()},
                "sri_duals": {s: c.FarkasDual for s, c in self.active_sri_cuts.items()},
                "edge_clique_duals": {e: c[0].FarkasDual for e, c in self.active_edge_clique_cuts.items()},
            }
            return -float("inf"), {}
        except AttributeError:
            logger.warning("FarkasDual extraction failed. Ensure InfUnbdInfo=1 is set.")
            return -float("inf"), {}

    def _extract_duals(self) -> None:
        """Extract dual values for all constraints in an optimal LP solution.

        Dual Extraction & Interpretation:
        - Mandatory Nodes (== 1): π_i is the shadow price.
        - Optional Nodes (<= 1): π_j >= 0.
        - Vehicle Limit (<= K): π_K >= 0 (fleet opportunity cost).
        - Cuts (Σ aλ <= RHS): Dual weights used to penalize "violated" routes in pricing.
        """
        self.dual_node_coverage = {}
        for node in range(1, self.n_nodes + 1):
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    # Set Partitioning (==) allows unrestricted duals. Set Packing (<=) restricts to >= 0.
                    if node in self.mandatory_nodes and self.strict_set_partitioning:
                        self.dual_node_coverage[node] = constr.Pi
                    else:
                        self.dual_node_coverage[node] = max(0.0, constr.Pi)

        self.dual_vehicle_limit = 0.0
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_vehicle_limit = max(0.0, constr.Pi)

        # Extract active cut duals
        self.dual_capacity_cuts = {s: abs(c.Pi) for s, c in self.active_capacity_cuts.items()}
        self.dual_sri_cuts = {s: abs(c.Pi) for s, c in self.active_sri_cuts.items()}
        self.dual_lci_cuts = {s: max(0.0, c.Pi) for s, c in self.active_lci_cuts.items()}
        self.dual_edge_clique_cuts = {e: max(0.0, c[0].Pi) for e, c in self.active_edge_clique_cuts.items()}

        if self.enable_dual_smoothing:
            self._apply_dual_smoothing()

    def _apply_dual_smoothing(self) -> None:
        """
        Apply Exponential Dual Smoothing to stabilize CG price signals.
        pi_smoothed = alpha * pi_current + (1 - alpha) * pi_prev

        WARNING: Smoothing modifies self.dual_node_coverage in-place.
        When active, the duals passed to pricing are NOT the true LP duals.
        This means:
        - The Lagrangian upper bound (obj_val + K * max_rc) is not valid.
        - CG convergence (added == 0) does not prove LP optimality.
        Only enable this flag when running in heuristic (non-exact) mode.

        Reference: Wentges (1997), Guyenne et al. (1994).
        """
        alpha = self.dual_smoothing_alpha
        for node, val in self.dual_node_coverage.items():
            prev = self.prev_dual_node_coverage.get(node, val)
            self.dual_node_coverage[node] = alpha * val + (1.0 - alpha) * prev
        self.prev_dual_node_coverage = self.dual_node_coverage.copy()

        prev_limit = self.prev_dual_vehicle_limit
        self.dual_vehicle_limit = alpha * self.dual_vehicle_limit + (1.0 - alpha) * prev_limit
        self.prev_dual_vehicle_limit = self.dual_vehicle_limit

    def solve_ip(self) -> Tuple[float, List[Route]]:
        """
        Solve the integer programme at the current B&B node.

        Re-imposes binary integrality on all λ variables and calls Gurobi's
        MIP solver.  Artificial variables are left continuous (they will be
        forced to zero by the Big-M penalty if any feasible integer solution
        exists).

        Returns:
            Tuple of:
                obj_value       – IP objective value.
                selected_routes – Routes with λ_k = 1 in the optimal solution.

        Raises:
            ValueError: If build_model() has not been called.
            RuntimeError: If the MIP solve fails.
        """
        if self.model is None:
            raise ValueError("Model not built.")
        if self.model.NumVars == 0:
            return 0.0, []

        # Enforce Integrality
        for var in self.lambda_vars:
            var.VType = GRB.BINARY
        self.model.update()

        try:
            self.model.optimize()
            if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                raise RuntimeError(f"IP solve failed: {self.model.Status}")
            return self.model.ObjVal, [self.routes[i] for i, v in enumerate(self.lambda_vars) if v.X > 0.5]
        finally:
            # Revert to continuous for further CG iterations
            for var in self.lambda_vars:
                var.VType = GRB.CONTINUOUS
            self.model.update()

    def add_route(self, route: Route) -> None:
        """
        Add a route (column) to the master problem.

        If the Gurobi model has already been built, the column is inserted
        into the live model immediately.  Otherwise it is buffered in
        self.routes and will be included when build_model() is called.

        Args:
            route: Route to insert as a new λ variable.
        """
        self.routes.append(route)
        if self.model is not None:
            self._add_column_to_model(route)

    def _add_column_to_model(self, route: Route) -> None:
        """
        Dynamically inserts a new column (route) into the Restricted Master Problem.
        This must inject the variable into both the base constraints AND any active cuts.

        Args:
            route: Route whose λ variable is being added.
        """
        if self.model is None:
            return

        var = self.model.addVar(obj=route.profit, vtype=GRB.BINARY, name=f"route_{len(self.lambda_vars)}")

        # Base Coverage Constraints
        for node in route.node_coverage:
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        # Base Vehicle Limit
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        self.lambda_vars.append(var)

        # Cut Plane Injection
        self._wire_route_into_active_cuts(route, var)
        self.model.update()

    def _wire_route_into_active_cuts(self, route: Route, var: Any) -> None:
        """
        Calculates the exact coefficient of the new route for every active cutting plane,
        guaranteeing that newly priced columns strictly respect previously separated inequalities.

        This ensures that columns added after some cuts have been discovered
        (standard Column Generation + Cutting Planes sequence) correctly
        participate in those cuts, maintaining LP relaxation validity.
        """
        if self.model is None:
            return

        full_path = [0] + route.nodes + [0]

        # 1. Capacity Cuts (RCCs) & SECs
        # The coefficient is the number of times the route crosses the cut boundary δ(S)
        for cut_dict in [self.active_capacity_cuts, self.active_sec_cuts, self.active_sec_cuts_local]:
            for node_set, constr in cut_dict.items():
                crossings = sum(
                    1 for i in range(len(full_path) - 1) if (full_path[i] in node_set) != (full_path[i + 1] in node_set)
                )
                if crossings > 0:
                    self.model.chgCoeff(constr, var, float(crossings))

        # 2. Subset Row Inequalities (SRIs)
        # The coefficient is ⌊|route ∩ S| / 2⌋
        for subset, constr in self.active_sri_cuts.items():
            intersection_size = sum(1 for n in subset if n in route.node_coverage)
            coeff = intersection_size // 2
            if coeff > 0:
                self.model.chgCoeff(constr, var, float(coeff))

        # 3. Localized Capacity Inequalities (LCIs)
        for cover_set, constr in self.active_lci_cuts.items():
            arc = self.active_lci_arcs.get(cover_set)
            if arc is not None:
                # Arc-saturation LCI: coefficient is 1 if route traverses arc
                contains_arc = any(
                    full_path[i] == arc[0] and full_path[i + 1] == arc[1] for i in range(len(full_path) - 1)
                )
                if contains_arc:
                    self.model.chgCoeff(constr, var, 1.0)
            else:
                # Node-based LCI: sum over node alphas
                node_alphas_for_set = self.active_lci_node_alphas.get(node_set, {})
                alpha_k = sum(
                    node_alphas_for_set.get(n, 1.0 if n in node_set else 0.0) for n in route.node_coverage if n != 0
                )
                if alpha_k > 1e-6:
                    self.model.chgCoeff(constr, var, alpha_k)

        # 4. Edge Clique Cuts
        for edge_tuple, (constr, _) in self.active_edge_clique_cuts.items():
            contains_edge = any(
                tuple(sorted((full_path[i], full_path[i + 1]))) == edge_tuple for i in range(len(full_path) - 1)
            )
            if contains_edge:
                self.model.chgCoeff(constr, var, 1.0)

    def purge_useless_columns(self, tolerance: float = -0.1) -> int:
        """
        Remove non-basic columns with significantly negative reduced cost.

        In a maximization LP, non-basic columns have RC <= 0. If a column is
        not in the basis and its reduced cost is highly negative, it is
        unlikely to enter the basis soon. Removing such columns prevents
        basis matrix bloat and keeps the RMP solver fast.

        Args:
            tolerance: Reduced cost threshold. Columns with RC < tolerance
                that are non-basic will be removed.

        Returns:
            Number of columns purged from the Gurobi model.
        """
        if not self.column_deletion_enabled or self.model is None or self.model.Status != GRB.OPTIMAL:
            return 0

        to_remove = []
        try:
            for i, v in enumerate(self.lambda_vars):
                if v.VBasis != 0 and tolerance > v.RC:
                    to_remove.append(i)
        except Exception:
            return 0

        if not to_remove:
            return 0

        for i in sorted(to_remove, reverse=True):
            self.global_column_pool.append(self.routes[i])
            self.model.remove(self.lambda_vars[i])
            self.lambda_vars.pop(i)
            self.routes.pop(i)

        self.model.update()
        return len(to_remove)

    def get_reduced_cost_coefficients(self) -> Dict[str, Any]:
        """
        Return the dual-value coefficients used by the pricing subproblem.

        Sign Convention & Maximization:
            The VRPP is formulated as a maximization problem. Under Gurobi's
            standard dual representation, the reduced cost calculation for
            pricing (searching for columns to boost the objective) is:
            rc(r) = profit(r) - Σ (a_{ik} * π_i), where π_i are the duals.
            - For >= constraints (e.g. RCC), Pi is non-positive; we use -Pi.
            - For <= constraints (e.g. SRI, Vehicle Limit), Pi is non-negative; we use +Pi.

        Returns:
            Dictionary mapping node ID (int) → dual value, plus the key
            "vehicle_limit" → vehicle-limit dual if a fleet cap is active.
        """
        node_duals: Dict[Union[int, str, frozenset[int], Tuple[int, int]], float] = {
            k: v for k, v in self.dual_node_coverage.items()
        }
        if self.vehicle_limit is not None:
            node_duals["vehicle_limit"] = self.dual_vehicle_limit

        return {
            "node_duals": node_duals,
            "rcc_duals": self.dual_capacity_cuts,
            "sri_duals": self.dual_sri_cuts,
            "edge_clique_duals": self.dual_edge_clique_cuts,
            "lci_duals": self.dual_lci_cuts,
            "lci_node_alphas": self.active_lci_node_alphas,
            "lci_arcs": self.active_lci_arcs,
        }

    def get_node_visitation(self) -> Dict[int, float]:
        """
        Aggregate fractional node-visitation values from the current LP solution.

        For each node i, returns Σ_{k: i ∈ route_k} λ_k, i.e. the total
        fractional "coverage" of that node across all selected routes.

        Returns:
            Mapping from node ID to aggregated λ coverage, or empty dict if
            the model has not been solved yet.
        """
        if self.model is None:
            return {}
        try:
            visits: Dict[int, float] = {}
            for i, v in enumerate(self.lambda_vars):
                if v.X > 1e-6:
                    for n in self.routes[i].node_coverage:
                        visits[n] = visits.get(n, 0.0) + v.X
            return visits
        except Exception:
            return {}

    def get_edge_usage(self, only_elementary: bool = False) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional edge visitation values from current LP solution.

        For an edge (i, j), returns Σ_{k: (i,j) ∈ route_k} λ_k.
        Edges are returned as canonical sorted tuples (min, max).

        Args:
            only_elementary: If True, only elementary routes are considered.

        Returns:
            Mapping from (u, v) -> fractional visitation sum.

        Note:
            Exact separation engines (e.g., Rounded Capacity Cuts) mathematically
            require flow conservation on a support graph of elementary routes.
            This filter prevents cyclic ng-routes from breaking max-flow separation.
        """
        if self.model is None:
            return {}

        edge_usage: Dict[Tuple[int, int], float] = {}
        try:
            for idx, var in enumerate(self.lambda_vars):
                val = var.X
                if val > 1e-6:
                    route = self.routes[idx]

                    # Task 2: Filter cyclic support for exact separation
                    if only_elementary and len(set(route.nodes)) != len(route.nodes):
                        continue

                    nodes = [0] + route.nodes + [0]
                    for i in range(len(nodes) - 1):
                        u, v = nodes[i], nodes[i + 1]
                        edge = (min(u, v), max(u, v))
                        edge_usage[edge] = edge_usage.get(edge, 0.0) + val
            return edge_usage
        except Exception:
            return {}

    def save_basis(self) -> Optional[Tuple[List[int], List[int]]]:
        """
        Save the basis (VBasis, CBasis) for variables and constraints.

        Returns:
            Tuple of (variable_basis, constraint_basis) lists, or None if model
            is not optimal or not initialized.
        """
        if self.model is None or self.model.Status != GRB.OPTIMAL:
            return None
        try:
            vbasis = [v.VBasis for v in self.model.getVars()]
            cbasis = [c.CBasis for c in self.model.getConstrs()]
            return (vbasis, cbasis)
        except Exception:
            return None

    def restore_basis(self, vbasis: List[int], cbasis: List[int]) -> None:
        """
        Restore the basis to warm-start the simplex algorithm.

        This robust version handles partial restorations in case of
        column-pool size mismatches across B&B nodes.

        Args:
            vbasis: Variable basis list.
            cbasis: Constraint basis list.
        """
        if self.model is None or vbasis is None or cbasis is None:
            return

        # Partial restore robust to column-pool mismatch
        vars_ = self.model.getVars()
        constrs_ = self.model.getConstrs()

        n_vars_saved = len(vbasis)
        n_constrs_saved = len(cbasis)
        n_vars_now = len(vars_)
        n_constrs_now = len(constrs_)

        if n_vars_now != n_vars_saved or n_constrs_now != n_constrs_saved:
            logger.debug(
                f"restore_basis: size mismatch — saved ({n_vars_saved} vars, "
                f"{n_constrs_saved} constrs) vs current ({n_vars_now} vars, "
                f"{n_constrs_now} constrs). Partial restore; warm-start may be degraded."
            )

        n_vars = min(n_vars_saved, n_vars_now)
        n_constrs = min(n_constrs_saved, n_constrs_now)

        for i in range(n_vars):
            vars_[i].VBasis = vbasis[i]

        for i in range(n_constrs):
            constrs_[i].CBasis = cbasis[i]

        self.model.update()
