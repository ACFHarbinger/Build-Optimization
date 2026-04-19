"""
Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for VRPP-specific valid inequalities:
- RCC (Rounded Capacity Cuts): Standard VRP cuts derived from bin-packing
  requirements (Lysgaard et al. 2004). Essential for problems with fractional
  load coverage.
- 3-SRI (Subset-Row Inequalities): Tightens the set partitioning relaxation
  for subsets of 3 nodes (Jepsen et al. 2008). These handle the "entanglement"
  of fractional routes that partially cover the same customer set.
- Fleet Cover: 0-1 knapsack covers derived from the global vehicle fleet
  limit (adapted from Balas 1975). These strengthen the fleet-size knapsack
  by lifting minimal covers on the route-selection variables.
- SEC (Subtour Elimination Cuts): Enforces connectivity for prize-collecting
  variants where the route must be a simple cycle connected to the depot.

Methodology:
    Each engine implements a `separate_and_add_cuts` method that inspects the
    fractional LP solution of the Master Problem, identifies violated
    inequalities from its specific family, and registers them into the MP's
    Gurobi model. The dual variables from these cuts are then passed back to
    the RCSPP pricing solver.

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
    - Jepsen, M., Petersen, B., Spoorendonk, S., & Pisinger, D. (2008).
      "Subset-row inequalities applied to the vehicle-routing problem with time windows."
      Operations Research, 56(2), 497-511.
    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164 (Cover inequalities).
"""

import hashlib
import itertools
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
    VRPPMasterProblem,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class CuttingPlaneEngine(ABC):
    """
    Abstract base class for cutting plane separation engines.

    Each engine implements a specific family of valid inequalities and
    provides separation algorithms to identify violated cuts.

    Implementations must only add globally valid inequalities. Cuts that
    are valid only at the current B&B node (e.g. cuts that use branching
    constraint data as parameters) must not be registered in the global
    cut pool.
    """

    @abstractmethod
    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify violated cuts and add them to the master problem.

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Engine-specific parameters

        Returns:
            Number of cuts added
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the engine name for logging."""
        pass

    @property
    def engines(self) -> List["CuttingPlaneEngine"]:
        """
        Return the list of sub-engines if this is a composite engine.
        Returns an empty list for simple engines.
        """
        return []

    def _is_orthogonal(self, candidate_vec: np.ndarray, active_vecs: List[np.ndarray], threshold: float = 0.8) -> bool:
        """
        Task 9 (SOTA): Cut Orthogonality filtering.
        Calculates maximum cosine similarity between candidate vector and active vectors.
        Rejects if similarity > threshold (i.e. if cut is too parallel to existing cuts).
        """
        if not active_vecs:
            return True

        norm_c = np.linalg.norm(candidate_vec)
        if norm_c < 1e-6:
            return False

        for a_vec in active_vecs:
            norm_a = np.linalg.norm(a_vec)
            if norm_a < 1e-6:
                continue
            cos_sim = np.abs(np.dot(candidate_vec, a_vec)) / (norm_c * norm_a)
            if cos_sim > threshold:
                return False
        return True


class RoundedCapacityCutEngine(CuttingPlaneEngine):
    r"""
    Rounded Capacity Cut (RCC) separation engine for VRPP Set Partitioning.

    Derived from the physical payload capacity (Q) of the fleet, RCCs are a
    standard family of valid inequalities for the Capacitated Vehicle Routing
    Problem (CVRP), adapted here for the VRPP Set Partitioning context.
    They enforce that the total flow crossing the boundary of a subset of nodes
    S must be sufficient to cover the total demand of S.

    Mathematical Formulation:
    -------------------------
    For a cluster of nodes S ⊆ N \ {0}, the RCC is defined as:
        ∑_{e ∈ δ(S)} x_e ≥ 2 * k(S)

    where:
    - δ(S) = {(i, j) ∈ E : i ∈ S, j ∉ S or i ∉ S, j ∈ S} is the edge cutset of S.
    - डिमांड(S) = ∑_{i ∈ S} q_i is the total demand requirement in S.
    - k(S) = ⌈ डिमांड(S) / Q ⌉ is the minimum number of vehicles required to
      service all customers in S given the capacity limit Q.
    - x_e = ∑_{k: e ∈ route_k} λ_k is the aggregated flow of edge e in the RMP solution.

    Theoretical Context:
    --------------------
    RCCs are polyhedrally stronger than simple connectivity constraints because
    they incorporate the physical resource constraint (Q). Since VRPP involves
    selecting nodes to visit, the Master Problem uses Set Partitioning, making
    RCCs essential for pruning fractional solutions where multiple routes
    "fractionally" visit nodes in a cluster without collectively satisfying
    the capacity logic.

    Separation Methodology:
    -----------------------
    This engine utilizes the separation framework of Lysgaard et al. (2004),
    supporting both heuristic (Tabu Search) and exact (Max-Flow Min-Cut)
    separation procedures. The separation operates on the support graph
    defined by edges with fractional flow x_e > 0.

    Integration with Pricing:
    --------------------------
    Dual variables associated with RCCs (dual_S) contribute to the reduced
    cost of labels in the RCSPP pricing subproblem. For every edge e in the
    boundary δ(S), the dual value π_S is subtracted from the edge cost.

    References:
    -----------
    Lysgaard et al. (2004), Section 3.2: "Capacity cuts"
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """
        Initialize the RCC separation engine.

        Args:
            v_model: VRPP model for edge indexing and demand tracking
            sep_engine: Separation engine from branch-and-cut module
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate RCCs and add to master problem.

        Algorithm:
        1. Get current edge usage from master problem's LP solution
           (Elementary only: cycles break max-flow conservation).
        2. Map edge usage to the separation engine's representation
        3. Run separation algorithms (heuristic + exact)
        4. Add violated cuts to master with Set Packing relaxation

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Unused (for interface consistency)

        Returns:
            Number of cuts added
        """
        # Task 2: Exact separation requires flow conservation on elementary routes.
        # Use filtered edge usage to prevent invalid cut generation from cyclic support.
        edge_vars = master.get_edge_usage(only_elementary=True)
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Build y_vals: fractional node visitation probabilities from the LP solution.
        # y_vals[i-1] = Σ_{k: i ∈ route_k} λ_k for each customer node i.
        # Index 0 corresponds to customer node 1 (depot node 0 is excluded).
        node_visits = master.get_node_visitation()
        n_customers = self.v_model.n_nodes - 1  # exclude depot
        y_vals = np.zeros(n_customers)
        for node, val in node_visits.items():
            if 1 <= node <= n_customers:
                y_vals[node - 1] = min(val, 1.0)  # clamp to [0,1] for LP fractional values

        # Separate cuts using the existing separation engine
        # BPC separation always operates on fractional LP solutions.
        # Use separate_fractional() for exact max-flow based RCC separation.
        node_depth = kwargs.get("node_depth", 0)
        violated_ineqs = self.sep_engine.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=max_cuts, node_count=node_depth
        )
        added_cuts = 0

        for ineq in violated_ineqs:
            if isinstance(ineq, CapacityCut):
                node_set = list(ineq.node_set)

                # For Set Packing, add cut with boundary relaxation
                # The master problem handles the y_i visitation tracking
                if master.add_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1
            elif isinstance(ineq, PCSubtourEliminationCut):
                node_set = list(ineq.node_set)
                # Form 2.1 is always global. Form 2.3 with fractional RHS is local only.
                # getattr used for defensive runtime robustness.
                is_global = ineq.facet_form == "2.1" and not getattr(ineq, "local_only", False)
                if master.add_sec_cut(
                    node_set,
                    ineq.rhs,
                    cut_name=ineq.facet_form,
                    global_cut=is_global,
                    node_i=ineq.node_i,
                    node_j=ineq.node_j,
                ):
                    added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        return "rcc"


class SubsetRowCutEngine(CuttingPlaneEngine):
    """
    Subset-Row Inequality (SRI) separation engine.

    The Subset-Row Inequalities are a class of Chvátal-Gomory rank-1
    inequalities derived from the Set Partitioning Problem (SPP) formulation.
    They are essential for tightening the LP relaxation when solving VRPs
    with column generation.

    Mathematical Formulation (3-SRI):
    ---------------------------------
    For a subset S of exactly |S| = 3 customer nodes, the 3-SRI (with p=2)
    is defined as:
        ∑_{k} ⌊ 1/2 * |S ∩ route_k| ⌋ * λ_k ≤ 1

    Logic:
    If three routes each visit exactly two nodes of the triplet S, the sum
    of their fractional variables λ_k cannot exceed 1.0. If it does (e.g.,
    three routes with λ_k = 0.5), the solution is fractional and violated.

    Theoretical Context:
    --------------------
    SRIs handle the "entanglement" of fractional routes. By penalizing
    routes that share too many nodes in a specified triplet, they force
    the branch-and-bound tree toward integer solutions. This implementation
    follows the separation logic established by Jepsen et al. (2008).

    Separation Methodology:
    -----------------------
    Heuristic search identifies candidate triplets where the sum of visitation
    variables y_i = ∑_{k: i ∈ route_k} λ_k is high. We specifically target
    triplets forming triangles in the fractional support graph.

    Integration with Pricing:
    --------------------------
    Specifically for 3-SRIs, the dual variables π_SRI modify the RCSPP label
    transition. A labeling state must track how many nodes of S it has
    visited. When the count transitions from 1 to 2, the dual value π_SRI
    is subtracted from the path profit (or added to reduced cost).
    """

    def __init__(self, v_model: VRPPModel):
        self.v_model = v_model

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        if master.model is None or len(master.routes) < 2:
            return 0

        # Structural SRI Separation (Academic Standard):
        # Instead of a myopic visitation-based heuristic, we identify candidate
        # triplets (i, j, k) that "co-occur" in active fractional columns.
        # This focuses separation on subsets most likely to be violated by
        # the current RMP solution.

        # 1. Identify "fractional" routes (0 < λ_k < 1)
        fractional_routes = []
        for idx, var in enumerate(master.lambda_vars):
            val = var.X
            if 0.01 < val < 0.99:
                fractional_routes.append((master.routes[idx], val))

        if not fractional_routes:
            return 0

        # 2. Build co-occurrence connectivity
        # We look for nodes that appear together in fractional routes.
        # This forms the "structural entanglement" suggested by Jepsen et al. (2008).
        adjacency: Dict[int, Set[int]] = {}
        for route, _ in fractional_routes:
            nodes = list(route.node_coverage)
            for i, j in itertools.combinations(nodes, 2):
                adjacency.setdefault(i, set()).add(j)
                adjacency.setdefault(j, set()).add(i)

        # 3. Candidate Triplets: Nodes forming a triangle or near-triangle in the support graph.
        candidate_subsets: Set[FrozenSet[int]] = set()
        for i, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            for j, k in itertools.combinations(list(neighbors), 2):
                # Triplets that share at least two fractional support edges
                candidate_subsets.add(frozenset([i, j, k]))

        if not candidate_subsets:
            return 0

        added = 0
        # 4. Evaluate violation for structural candidates
        for node_fset in candidate_subsets:
            if added >= max_cuts:
                break
            if self._evaluate_and_add_sri(master, set(node_fset), **kwargs):
                added += 1

        return added

    def _evaluate_and_add_sri(self, master: VRPPMasterProblem, node_set: Set[int], **kwargs) -> bool:
        """
        Calculate SRI violation and add to master if violated.

        Task 9 (SOTA): Cut Similarity Filtering.
        Evaluates the violation and similarity of the proposed SRI to the
        active cut pool. Rejects if similarity > threshold.
        """
        val = 0.0
        coeff_dict: Dict[str, float] = {}
        threshold = kwargs.get("cut_orthogonality_threshold", 0.8)
        for idx, route in enumerate(master.routes):
            try:
                lam = master.lambda_vars[idx].X
            except Exception:
                continue

            if lam < 1e-6:
                continue

            # Fix 13: Key by content hash, not index, to remain robust against column purging.
            content = ",".join(map(str, route.nodes))
            route_h = hashlib.md5(content.encode()).hexdigest()

            count = len(node_set.intersection(route.node_coverage))
            coeff = count // 2
            if coeff > 0:
                coeff_dict[route_h] = float(coeff)
                val += float(coeff) * lam

        if val > 1.0 + 1e-4:
            # Task 9: Check similarity using content-mapped vectors
            active_sri_vecs = list(master.global_cut_pool.active_sri_vectors.values())
            if not self._is_orthogonal_content(coeff_dict, active_sri_vecs, threshold):
                return False

            return master.add_subset_row_cut(list(node_set))
        return False

    def _is_orthogonal_content(
        self, candidate_dict: Dict[str, float], active_vecs: List[Dict[str, float]], threshold: float = 0.8
    ) -> bool:
        """
        Fix 4: Content-keyed orthogonality check.
        """
        if not active_vecs:
            return True

        norm_c = np.sqrt(sum(v * v for v in candidate_dict.values()))
        if norm_c < 1e-6:
            return False

        for a_dict in active_vecs:
            dot = 0.0
            for h, v in candidate_dict.items():
                if h in a_dict:
                    dot += v * a_dict[h]
            norm_a = np.sqrt(sum(v * v for v in a_dict.values()))
            if norm_a < 1e-6:
                continue
            cos_sim = abs(dot) / (norm_c * norm_a)
            if cos_sim > threshold:
                return False
        return True

    def get_name(self) -> str:
        return "sri"


class EdgeCliqueCutEngine(CuttingPlaneEngine):
    """
    Fleet Cover Inequality (Edge-Clique) separation engine.

    This engine separates valid inequalities derived from conflict graphs
    on the route-selection variables. Specifically, it identifies cliques
    in the conflict graph of routes that compete for a shared resource
    (in this case, single-use of a directed arc or a total vehicle count).

    Mathematical Formulation:
    -------------------------
    For a physical edge e = (i, j), let C(e) be the set of all routes in
    the RMP that traverse edge e. The Edge-Clique inequality is:
        ∑_{k ∈ C(e)} λ_k ≤ 1

    Theoretical Context:
    --------------------
    Standard Set Partitioning restricts each node to one visit, but
    Edge-Clique inequalities extend this to the "edges" of the graph.
    They are derived by lifting minimal covers on the arc capacity knapsack,
    forming facets of the resulting knapsack polytope. This implementation
    adapts the Lifting and Cover Inequality (LCI) logic of Balas (1975)
    and Barnhart et al. (2000).

    Heuristic Separation:
    --------------------
    The engine computes the aggregated flow x_{ij} for every edge in the
    current solution. Edges where x_{ij} > 1.0 + ε are candidates for a
    clique cut. For anonymous fleets where route weights are uniform (=1),
    the lifting coefficient for all routes in the clique simplifies to 1.0.

    Operational Logic:
    ------------------
    1. Scan the Restricted Master Problem columns (λ_k).
    2. Build an edge-indexed map of crossing routes.
    3. Identify "oversaturated" edges where fractional routes sum to > 1.0.
    4. Register the clique constraint to prune the fractional space.
    """

    def __init__(self, v_model: VRPPModel, capacity: float = 1.0, epsilon: float = 0.01) -> None:
        """
        Initialize the LCI separation engine.

        Args:
            v_model: VRPP model for edge indexing.
            capacity: Physical capacity of the arc (default 1.0 for VRP).
            epsilon: Violation threshold.
        """
        self.v_model = v_model
        self.capacity = capacity
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """
        Identify violated LCIs by lifting minimal covers.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        # Edge-clique cuts are globally valid: they depend only on the edge
        # capacity structure, not on branching decisions. Safe for GlobalCutPool.

        # 1. Aggregate edge flow: x_e = Σ_{k} w_{ek} λ_k
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        edge_to_routes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # arc -> list of (route_idx, weight)

        for ridx in route_values:
            route = master.routes[ridx]
            nodes = [0] + route.nodes + [0]
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                e = tuple(sorted((u, v)))
                # Track weights (crossings) for the knapsack formulation
                # In standard VRP, weight is 1. If route repeats arc, weight > 1.
                found = False
                if e in edge_to_routes:
                    for j, (ridx_existing, w) in enumerate(edge_to_routes[e]):  # type: ignore[index]
                        if ridx_existing == ridx:
                            edge_to_routes[e][j] = (ridx_existing, w + 1)  # type: ignore[index]
                            found = True
                            break
                if not found:
                    edge_to_routes.setdefault(e, []).append((ridx, 1))  # type: ignore[arg-type]

        # 2. Identify saturated edges
        saturated_edges = []
        for e, routes_info in edge_to_routes.items():
            flow = sum(master.lambda_vars[ridx].X * w for ridx, w in routes_info)
            if flow > self.capacity + self.epsilon:
                saturated_edges.append((e, routes_info, flow))

        # Sort by flow violation
        saturated_edges.sort(key=lambda x: x[2], reverse=True)

        added_cuts = 0
        for edge_tuple, routes_info, _ in saturated_edges:
            if added_cuts >= max_cuts:
                break

            # 3. Find minimal cover C
            # Sort routes by λ_k descending
            routes_info.sort(key=lambda x: master.lambda_vars[x[0]].X, reverse=True)

            cover_indices = []
            weight_sum = 0
            for idx, w in routes_info:
                cover_indices.append((idx, w))
                weight_sum += w
                if weight_sum > self.capacity:
                    break

            if weight_sum <= self.capacity:
                continue  # Should not happen for saturated arc if all w >= 1

            # 4. O(1) Clique Lifting
            # For Set Partitioning, capacity is 1. Any pair of routes traversing
            # this edge form a clique. The lifting coefficient is uniformly 1.0.
            rhs = 1.0
            coefficients = {ridx: 1.0 for ridx, _ in routes_info}

            # 5. Add to master
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1], coefficients=coefficients, rhs=rhs):
                added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        return "edge_clique"


class KnapsackCoverEngine(CuttingPlaneEngine):
    """
    Separates Cover Inequalities for knapsack constraints in the Master Problem.

    In the VRPP context, this primarily targets the fleet size (vehicle limit)
    constraint (Σ λ_k ≤ K) or any other resource knapsacks. While simpler
    than a general knapsack with varied weights, cover cuts help strengthen
    the integrality of the master problem by excluding fractional solutions
    that exceed the logical resource boundary.

    Mathematical Formulation:
    -------------------------
    Given a knapsack constraint ∑ w_j x_j ≤ b, a set of routes C is a 'cover'
    if ∑_{j ∈ C} w_j > b. The resulting cover inequality is:
        ∑_{j ∈ C} λ_j ≤ |C| - 1

    Theoretical Rationale:
    ----------------------
    A cover inequality excludes the point where all routes in C are selected
    with fractional values that sum to more than the capacity b. For the
    fleet constraint (w_j = 1), a minimal cover is any subset C of size K+1.
    This implementation adapts the logic of Barnhart et al. (2000).

    Interaction with Master Problem:
    --------------------------------
    If a vehicle_limit (K) is explicitly set in the RMP, the fleet constraint
    already prevents the LP from selecting more than K total routes.
    However, cover cuts provide additional polyhedral strength by
    specifically targeting the most fractional routes in the solution.

    Reference: Barnhart et al. (2000), Section 5.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify and add violated cover inequalities for the vehicle limit.
        """
        # Task 4 & 6: LCI already handles fleet-size tightening more effectively.
        # This engine serves as a fallback or for non-fleet knapsacks.
        if master.vehicle_limit is None or master.model is None:
            return 0

        # Σ λ_k ≤ K.
        # A cover C is any subset of routes such that |C| > K.
        # The inequality is Σ_{k ∈ C} λ_k ≤ |C| - 1.
        # Violation occurs if Σ_{k ∈ C} λ_k > |C| - 1.

        # Strategy: find routes with largest fractional values.
        # If the sum of the top K+1 λ_k values exceeds K, we have a violation.
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        if len(route_values) <= master.vehicle_limit:
            return 0

        sorted_indices = sorted(route_values.keys(), key=lambda i: route_values[i], reverse=True)
        top_k_plus_1 = sorted_indices[: master.vehicle_limit + 1]
        sum_top = sum(route_values[i] for i in top_k_plus_1)

        if sum_top > master.vehicle_limit + 1e-4:
            # Violated cover found.
            # The cover is the set of nodes visited by the top K+1 routes.
            # Use their union as the node set S for the capacity cut registry.
            cover_node_set: set = set()
            for i in top_k_plus_1:
                cover_node_set.update(master.routes[i].node_coverage)
            cover_node_set.discard(0)  # exclude depot

            if cover_node_set and master.add_capacity_cut(list(cover_node_set), float(master.vehicle_limit)):
                return 1

        return 0

    def get_name(self) -> str:
        return "cover"


class CompositeCuttingPlaneEngine(CuttingPlaneEngine):
    """
    Combines multiple separation engines to run in sequence.
    """

    def __init__(self, engines: List[CuttingPlaneEngine]) -> None:
        self._engines = engines

    @property
    def engines(self) -> List[CuttingPlaneEngine]:
        return self._engines

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        per_engine = max(1, max_cuts // len(self.engines))
        total_added = 0
        for engine in self.engines:
            total_added += engine.separate_and_add_cuts(master, per_engine, **kwargs)
        return total_added

    def get_name(self) -> str:
        return "composite"


class BasicFleetCoverEngine(CuttingPlaneEngine):
    """
    Separates basic cover inequalities for the fleet-size knapsack.

    The fleet-size constraint Σλₖ ≤ K is a 0-1 knapsack where every route
    has weight 1. A cover C is any subset of routes with |C| > K. The cover
    inequality Σ_{k∈C} λₖ ≤ K is valid and violated when the top K+1
    fractional routes sum to more than K.

    This engine is a strict uniform knapsack cover. It does not implement
    true sequence-independent lifting because for a uniform knapsack,
    all lifting coefficients theoretically collapse to 1.0.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate and add basic fleet cover inequalities.

        Identifies a violated cover C (a set of routes with |C| > K whose
        fractional values sum to more than K) and adds the basic cover
        inequality Σ_{k∈C} λ_k ≤ K to the master problem.
        """
        if master.vehicle_limit is None or not master.lambda_vars:
            return 0

        K = float(master.vehicle_limit)

        # 1. Identify fractional routes sorted by λ_k descending.
        active_routes = []
        for i, var in enumerate(master.lambda_vars):
            try:
                val = var.X
            except Exception:
                continue
            if val > 1e-4:
                active_routes.append((i, val))
        active_routes.sort(key=lambda x: x[1], reverse=True)

        if len(active_routes) <= K:
            return 0

        # 2. Find a minimal cover C: the top K+1 routes by fractional value.
        cover_indices = [idx for idx, _ in active_routes[: int(K) + 1]]
        sum_val = sum(master.lambda_vars[i].X for i in cover_indices)

        if sum_val <= K + self.epsilon:
            return 0  # No violation — cut would not be useful.

        # 3. Basic uniform lifting
        # Because we have uniform route weights (=1), the exact lifting
        # coefficients alpha_k simplify back to 1.0 across the board.
        # We explicitly calculate it just as alpha_k = 1.0.
        coefficients = {i: 1.0 for i in cover_indices}
        for i in range(len(master.lambda_vars)):
            if i not in cover_indices:
                coefficients[i] = 1.0

        # Represent the cover by its union of visited customer nodes.
        # add_lci_cut keys by node_set; two covers with the same node union
        # are treated as equivalent.
        cover_node_set: List[int] = []
        seen: set = set()
        for i in cover_indices:
            for n in master.routes[i].node_coverage:
                if n not in seen:
                    cover_node_set.append(n)
                    seen.add(n)

        if not cover_node_set:
            return 0

        added = 0
        if master.add_lci_cut(cover_node_set, K, coefficients):
            added += 1
        return added

    def get_name(self) -> str:
        return "fleet_cover"


class PhysicalCapacityLCIEngine(CuttingPlaneEngine):
    """
    Separates Lifted Cover Inequalities (LCI) over node-visit knapsacks.

    Targets the heterogeneous physical capacity knapsack (Σ_{i ∈ S} q_i y_i ≤ Q),
    where q_i is the demand of customer i and Q is the vehicle capacity.
    These cuts target the visitation variables (y_i) rather than the route
    selection variables (λ_k).

    Mathematical Formulation:
    -------------------------
    1. Identify a minimal cover S such that ∑_{i ∈ S} q_i > Q.
    2. The basic cover inequality is ∑_{i ∈ S} y_i ≤ |S| - 1.
    3. Strengthen the inequality via sequence-independent lifting:
       ∑_{i ∈ S} α_i y_i + ∑_{j ∉ S} α_j y_j ≤ α_0

    Lifting Procedure (Balas 1975):
    -------------------------------
    Calculates exact lifting coefficients (α_i) for nodes based on their
    fractional demands q_i. These node-level coefficients are then aggregated
    into the route variables λ_k for the RMP:
        ∑_{k} (∑_{i ∈ node_coverage_k} α_i) λ_k ≤ α_0

    Operational Logic:
    ------------------
    - Sort nodes by visitation probability y_i to find heuristic covers.
    - Compute lifting coefficients using prefix sums of sorted demands.
    - Map node-alphas to route-coefficients and register the cut in the RMP.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        if master.model is None or not master.lambda_vars:
            return 0

        Q = self.v_model.capacity
        y_vals = master.get_node_visitation()

        added = 0
        already_covered: List[List[int]] = []

        # Try multiple cover orderings to find distinct violated LCIs.
        # Paper §6: Gu et al. (1995a) heuristic generates one LCI per saturated structure.
        # We diversify by sorting by three different keys so non-dominated covers are found.
        customer_nodes = [i for i in range(1, self.v_model.n_nodes) if y_vals.get(i, 0.0) > 1e-4]
        orderings = [
            sorted(customer_nodes, key=lambda i: y_vals.get(i, 0.0), reverse=True),  # high visitation first
            sorted(customer_nodes, key=lambda i: self.v_model.wastes.get(i, 0.0), reverse=True),  # heavy demand first
            sorted(
                customer_nodes, key=lambda i: y_vals.get(i, 0.0) * self.v_model.wastes.get(i, 0.0), reverse=True
            ),  # weighted first
        ]

        for nodes_sorted in orderings:
            if added >= max_cuts:
                break

            cover: List[int] = []
            weight_sum = 0.0

            for i in nodes_sorted:
                cover.append(i)
                weight_sum += self.v_model.wastes.get(i, 0.0)
                if weight_sum > Q:
                    break

            if weight_sum <= Q or not cover:
                continue

            # Deduplicate: skip if this cover is too similar to an already-added one
            cover_set = frozenset(cover)
            if any(frozenset(prev) == cover_set for prev in already_covered):
                continue

            cover_y_sum = sum(y_vals.get(i, 0.0) for i in cover)
            if cover_y_sum <= len(cover) - 1 + self.epsilon:
                continue

            # Sequence-independent lifting coefficients (Gu, Nemhauser, Savelsbergh 1995a)
            cover_weights_sorted = sorted([self.v_model.wastes.get(i, 0.0) for i in cover], reverse=True)

            prefix_sums = [0.0] * (len(cover) + 1)
            for idx, w in enumerate(cover_weights_sorted):
                prefix_sums[idx + 1] = prefix_sums[idx] + w

            def compute_lifting_coeff(
                wj: float,
                _ps: List[float] = prefix_sums,
                _cover_len: int = len(cover),
            ) -> float:
                # Gu et al. (1995a) exact lifting: α_j = max p such that w_j + Σ_{top p} ≥ Q+ε
                for p in range(_cover_len, -1, -1):
                    if wj + _ps[p] > Q:
                        return float(p)
                return 0.0

            # Node-level lifting coefficients α_i
            alpha_nodes: Dict[int, float] = {}
            for j in range(1, self.v_model.n_nodes):
                if j in cover:
                    alpha_nodes[j] = 1.0
                else:
                    alpha_nodes[j] = compute_lifting_coeff(self.v_model.wastes.get(j, 0.0))

            # Route-level lambda coefficients Σ_{i ∈ route ∩ domain} α_i
            # All routes must be included regardless of current LP value; a route
            # with var.X ≈ 0 now may become active in future LP iterations, and
            # omitting it leaves the LCI constraint under-constrained from the start.
            coefficients: Dict[int, float] = {}
            for k, route in enumerate(master.routes):
                alpha_k = sum(alpha_nodes.get(n, 0.0) for n in route.node_coverage if n != 0)
                if alpha_k > 1e-6:
                    coefficients[k] = alpha_k

            # Pass both route coefficients AND node alphas for pricing dual integration.
            if master.add_lci_cut(cover, float(len(cover) - 1), coefficients, node_alphas=alpha_nodes):
                already_covered.append(cover)
                added += 1

        return added

    def get_name(self) -> str:
        return "physical_lci"


class SaturatedArcLCIEngine(CuttingPlaneEngine):
    r"""
    Saturated-Arc Lifted Cover Inequality engine — the primary LCI mechanism
    described in Barnhart, Hane, and Vance (2000) §6.

    Algorithm (paper §6, Gu et al. 1995a):
    -----------------------------------------
    Finding the most violated Lifted Cover Inequality is theoretically NP-hard.
    Consequently, this module employs a heuristic mechanism to identify minimal
    covers derived from saturated arcs in the fractional support graph.

    For each ARC (i, j) that is saturated in the current LP solution
    (i.e., fractional arc flow x_{ij} > capacity − ε):

    1. Identify C(i,j) = {routes k : arc (i,j) ∈ route k}.
    2. If C(i,j) is a cover (Σ_{k ∈ C} demand_k > arc_capacity), form the
       basic cover inequality:
           Σ_{k ∈ C} λ_k ≤ |C| − 1
    3. Lift coefficients for routes NOT in C using the Gu et al. (1995a)
       exact sequence-independent procedure:
           α_k = max p s.t. p routes from \bar{C} can fit with remaining capacity.
    4. Translate to the path-based formulation via delta_{ij}^p indicators.
    5. Add violated LCI to the master.

    Pricing Integration (§4.2):
    In the VRPP RCSPP, when the DP traverses arc (i,j), the reduced cost is
    decreased by α_{ij}^k · γ_{ij}, exactly as in the paper.  Since arcs are
    unit-capacity in VRPP, α_{ij}^k = 1 for all routes in C, and the lifting
    coefficients for routes outside C are computed via the Gu heuristic.

    Arc Capacity in VRPP:
    ---------------------
    In Set Partitioning, each arc can appear in at most 1 selected route in any
    integer solution.  A fractional LP may assign more than 1.0 total flow to an
    arc across multiple fractional routes; this constitutes arc saturation.
    The corresponding capacity knapsack is:
        Σ_{k : arc ∈ route_k} λ_k ≤ 1   (arc capacity = 1 in unit-flow VRP)
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:  # noqa: C901
        """
        Separate saturated-arc LCIs and add to master.

        Returns:
            Number of cuts added.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        # 1. Aggregate arc flows x_{ij} = Σ_k λ_k · δ_{ij}^k
        arc_to_routes: Dict[Tuple[int, int], List[int]] = {}
        route_values: Dict[int, float] = {}

        for ridx, var in enumerate(master.lambda_vars):
            try:
                val = var.X
            except Exception:
                continue
            if val < 1e-6:
                continue
            route_values[ridx] = val
            route = master.routes[ridx]
            full_path = [0] + route.nodes + [0]
            for pos in range(len(full_path) - 1):
                arc = (full_path[pos], full_path[pos + 1])
                arc_to_routes.setdefault(arc, []).append(ridx)

        # 2. Find saturated arcs: Σ_{k : arc ∈ route_k} λ_k > 1.0 − ε
        arc_capacity = 1.0  # unit capacity in VRPP SP
        saturated: List[Tuple[float, Tuple[int, int]]] = []
        for arc, route_indices in arc_to_routes.items():
            flow = sum(route_values.get(k, 0.0) for k in route_indices)
            if flow > arc_capacity - self.epsilon:
                violation = flow - arc_capacity
                saturated.append((violation, arc))

        # Sort by most violated first — focus separation on the tightest constraints
        saturated.sort(reverse=True)

        added = 0
        for _violation, arc in saturated:
            if added >= max_cuts:
                break

            cover_indices = arc_to_routes[arc]
            if not cover_indices:
                continue

            # 3. Basic cover: all routes using this arc form a cover (sum > capacity = 1)
            cover_flow = sum(route_values.get(k, 0.0) for k in cover_indices)
            if cover_flow <= arc_capacity - self.epsilon:
                continue  # Not actually violated

            # 4. Sequence-independent lifting (Gu et al. 1995a)
            # For arc capacity = 1, weights are all 1.0 (each route contributes 1).
            # α_k = 1 for routes in C; for routes NOT in C, α_k = 1 if adding that
            # route saturates the arc with the current cover.
            cover_set_idx = set(cover_indices)
            n_cover = len(cover_indices)
            rhs = float(n_cover - 1)

            # Coefficient for routes IN cover: 1.0
            coefficients: Dict[int, float] = {k: 1.0 for k in cover_indices}

            # Coefficient for routes NOT in cover: α_k ∈ {0, 1}
            # Since arc capacity = 1 and cover already saturates it, any additional route
            # also using this arc has lifting coefficient α_k = 1 (extends the cover).
            for ridx_other in route_values:
                if ridx_other in cover_set_idx:
                    continue
                route_other = master.routes[ridx_other]
                full_path = [0] + route_other.nodes + [0]
                if any(full_path[pos] == arc[0] and full_path[pos + 1] == arc[1] for pos in range(len(full_path) - 1)):
                    coefficients[ridx_other] = 1.0

            # Node-level alphas: α_i = 1 for nodes in routes that use this arc.
            # These allow pricing to penalise arc-covering routes at the node level.
            alpha_nodes: Dict[int, float] = {}
            for k in cover_indices:
                for node in master.routes[k].node_coverage:
                    alpha_nodes[node] = max(alpha_nodes.get(node, 0.0), 1.0)

            # 5. Check current violation before adding
            lhs_val = sum(coefficients.get(k, 0.0) * route_values.get(k, 0.0) for k in coefficients)
            if lhs_val <= rhs + 1e-4:
                continue  # Not violated in fractional solution

            # Use the covered node set as the LCI dedup key.
            # This is semantically correct: the cut constrains routes that share
            # this node cover, and two arcs with identical covered-node sets
            # would generate the same constraint (so skipping is safe).
            cover_node_list = sorted(alpha_nodes.keys())
            if not cover_node_list:
                continue

            # Pass the saturated arc so the pricing DP gates the dual on exact arc
            # traversal (i→j) rather than on any visit to a node in the cover set.
            # This implements Barnhart et al. (2000) §4.2 c'_lm^k = c_lm^k + α·γ
            # without the over-penalisation caused by a node-visit approximation.
            if master.add_lci_cut(
                cover_node_list,
                rhs,
                coefficients,
                node_alphas=alpha_nodes,
                arc=arc,
            ):
                added += 1

        return added

    def get_name(self) -> str:
        return "saturated_arc_lci"


def create_cutting_plane_engine(
    engine_name: str,
    v_model: VRPPModel,
    sep_engine: Optional[SeparationEngine] = None,
) -> CuttingPlaneEngine:
    """
    Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc", "src", "cover", or "all")
        v_model: VRPP model for problem data
        sep_engine: Separation engine (required for RCC/SRC)

    Returns:
        Instance of the requested cutting plane engine

    Raises:
        ValueError: If engine_name is not recognized or required args missing

    Example:
        >>> sep_engine = SeparationEngine(v_model)
        >>> cut_engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
        >>> cuts_added = cut_engine.separate_and_add_cuts(master, max_cuts=5)
    """
    if engine_name == "rcc":
        if sep_engine is None:
            raise ValueError("RCC engine requires sep_engine parameter")
        return RoundedCapacityCutEngine(v_model, sep_engine)
    elif engine_name == "sri":
        return SubsetRowCutEngine(v_model)
    elif engine_name == "edge_clique":
        return EdgeCliqueCutEngine(v_model)
    elif engine_name == "fleet_cover":
        return BasicFleetCoverEngine(v_model)
    elif engine_name == "physical_lci":
        return PhysicalCapacityLCIEngine(v_model)
    elif engine_name == "saturated_arc_lci":
        return SaturatedArcLCIEngine(v_model)
    elif engine_name == "cover":
        if sep_engine is None:
            raise ValueError("Cover engine requires sep_engine parameter")
        return KnapsackCoverEngine(v_model, sep_engine)
    elif engine_name in ("all", "composite"):
        engines = []
        if sep_engine is not None:
            engines.append(RoundedCapacityCutEngine(v_model, sep_engine))
        engines.append(SubsetRowCutEngine(v_model))  # type: ignore[arg-type]
        # Edge clique is always added as a spatial base cut
        engines.append(EdgeCliqueCutEngine(v_model))  # type: ignore[arg-type]
        engines.append(BasicFleetCoverEngine(v_model))  # type: ignore[arg-type]
        engines.append(PhysicalCapacityLCIEngine(v_model))  # type: ignore[arg-type]
        engines.append(SaturatedArcLCIEngine(v_model))  # type: ignore[arg-type]
        if sep_engine is not None:
            engines.append(KnapsackCoverEngine(v_model, sep_engine))  # type: ignore[arg-type]

        comp = CompositeCuttingPlaneEngine(engines)  # type: ignore[arg-type]
        return comp

    else:
        valid = [
            "rcc",
            "sri",
            "edge_clique",
            "fleet_cover",
            "physical_lci",
            "saturated_arc_lci",
            "cover",
            "all",
            "composite",
        ]
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: {valid}")
