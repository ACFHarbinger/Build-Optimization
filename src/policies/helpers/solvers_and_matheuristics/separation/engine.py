"""
Separation Algorithms for VRPP Cutting Planes.

Implements exact and heuristic separation procedures for identifying violated
valid inequalities in the VRPP linear programming relaxation.

References:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.

    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut algorithm
    for the capacitated vehicle routing problem". Mathematical Programming, 100(2), 423-445.

    Padberg, M., & Rinaldi, G. (1991). "A Branch-and-cut Algorithm for the Resolution of
    Large-scale Symmetric Traveling Salesman Problems". SIAM Review, 33(1), 60-100.

    - Prize-Collecting Subtour Elimination Constraints (PC-SEC) follow Fischetti et al. (1997)
    - Capacity Constraints and fractional separation follow Lysgaard et al. (2004)
    - Comb inequalities are disabled in favor of Gurobi's internal polyhedral cuts
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow

from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
    CapacityCut,
    CombInequality,
    Inequality,
    PCSubtourEliminationCut,
)


class SeparationEngine:
    """
    Separation algorithms for finding violated inequalities.

    Implements exact and heuristic separation procedures for identifying violated
    valid inequalities in the VRPP linear programming relaxation.

    Configuration:
        USE_COMB_CUTS: If True, enable custom comb inequality separation.
                       Default False - Gurobi's internal polyhedral cuts are preferred.

        enable_heuristic_rcc_separation: If True, enable (heuristic) fractional capacity separation.
                       Default False for instances with n > 75 (see CVRPSEP note below).

    Note:
        State-of-the-art implementations typically rely on shrinking heuristics (such as
        Lysgaard's CVRPSEP C++ library) to scale to N > 100.
    """

    # Global toggle for comb inequality separation
    # Set to False to disable custom comb cuts and rely on Gurobi's internal heuristics
    USE_COMB_CUTS = False

    # Task 12c: Period for running expensive exact separation at non-root nodes.
    # Set to 1 for exhaustive separation every iteration.
    _EXACT_SEP_PERIOD = 3

    def __init__(
        self,
        model,
        enable_heuristic_rcc_separation: bool = True,
        enable_comb_cuts: bool = False,
    ):
        """
        Initialize separation engine.

        Args:
            model: VRPPModel instance.
            enable_heuristic_rcc_separation: Whether to enable heuristic connected-component RCC separation.
        """
        self.model = model
        self.pool: List[Inequality] = []  # Pool of stored inequalities
        self.enable_heuristic_rcc_separation = enable_heuristic_rcc_separation
        self.enable_comb_cuts = enable_comb_cuts

    def separate_integer(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 100,
        iteration: int = 0,
        sec_only: bool = False,
    ) -> List[Inequality]:
        """
        Fast separation for integer solutions (MIPSOL callback).

        Uses efficient heuristic methods suitable for integer values:
        - Connected components for subtours (O(n²))
        - Demand-based clustering for capacity cuts (O(n²))

        Args:
            x_vals: Integer edge values from MIP solution.
            y_vals: Integer node visit values from MIP solution.
            max_cuts: Maximum number of cuts to return.
            iteration: Current iteration number.

        Returns:
            List of violated inequalities sorted by degree of violation.
        """
        self.pool = []

        # Step 1: Fast subtour separation using connected components
        self._separate_disconnected_components(x_vals, y_vals)

        # Step 2: Heuristic SEC separation (Kruskal-based GSEC_H2)
        if not sec_only:
            self._separate_gsec_h2(x_vals, y_vals)

        # Step 3: Procedure PCSEC_BUILD: Strengthen sets and select strongest facets
        self._strengthen_pool(self.pool, x_vals, y_vals)

        # Step 4: Filter and return most violated cuts
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]
        violated.sort()  # Sort by violation descending
        return violated[:max_cuts]

    def separate_fractional(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 50,
        iteration: int = 0,
        node_count: int = 0,
    ) -> List[Inequality]:
        """
        Exact separation for fractional LP nodes (MIPNODE callback).

        WARNING: This is a heuristic separation procedure based on connected
        components of the LP support graph. It finds violated RCCs only for
        subsets S that form a connected component. Violated RCCs for
        non-component subsets are missed. Exact RCC separation requires
        solving a max-flow problem per candidate set (see Padberg & Rinaldi 1991;
        Lysgaard et al. 2004). The enable_heuristic_rcc_separation parameter
        does NOT switch to exact separation — it enables this heuristic.
        """
        self.pool = []

        # Strategy: More aggressive separation at root node
        if node_count == 0:
            # Root node: Full exact separation
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)
            self._separate_gsec_h2(x_vals, y_vals)
            self._separate_pcsec_exact(x_vals, y_vals, root_node=True)

            # Exact fractional capacity separation (toggle-controlled)
            if self.enable_heuristic_rcc_separation:
                self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=True)
        else:
            # Shallow nodes: Limited exact separation
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)

            # Run exact separation less frequently
            if iteration % self._EXACT_SEP_PERIOD == 0:
                self._separate_gsec_h2(x_vals, y_vals)
                self._separate_pcsec_exact(x_vals, y_vals, root_node=False)

                # Exact fractional capacity separation (toggle-controlled)
                if self.enable_heuristic_rcc_separation:
                    self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=False)

        # Filter and return most violated cuts
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]

        # Procedure GSEC_BUILD: Strengthen sets and select strongest facets
        self._strengthen_pool(violated, x_vals, y_vals)

        violated.sort()  # Sort by violation descending
        return violated[:max_cuts]

    def separate(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 100,
        iteration: int = 0,
    ) -> List[Inequality]:
        """
        Main separation procedure for VRPP (legacy interface).

        This method is maintained for backward compatibility but is not used
        in the true Branch-and-Cut callback. Use separate_integer() or
        separate_fractional() instead.

        Args:
            x_vals: Fractional edge values from LP solution (indexed by edge list).
            y_vals: Fractional node visit values from LP solution (optional).
            max_cuts: Maximum number of cuts to return.
            iteration: Current iteration number (for adaptive strategies).

        Returns:
            List of violated inequalities sorted by degree of violation.
        """
        self.pool = []

        # Step 1: Separate subtour elimination constraints (SECs)
        self._separate_subtours_heuristic(x_vals, y_vals)

        # Step 2: Separate capacity cuts (heuristic and exact)
        self._separate_capacity_cuts(x_vals, y_vals)

        # Step 3: Exact subtour separation (max-flow based) - expensive, run periodically
        if iteration % 5 == 0:
            self._separate_pcsec_exact(x_vals, y_vals)
            self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals)

        # Step 4: Comb inequalities (heuristic) - advanced cuts
        # DISABLED: Gurobi's internal clique/cover cuts are preferred
        if self.USE_COMB_CUTS and iteration % 10 == 0:
            self._separate_comb_heuristic(x_vals, y_vals)

        # Filter and return most violated cuts
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]
        violated.sort()  # Sort by violation descending
        return violated[:max_cuts]

    def _separate_subtours_heuristic(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """
        Heuristic subtour elimination using connected components and greedy expansion.
        """
        self._separate_disconnected_components(x_vals, y_vals)
        self._separate_weak_subtours(x_vals, y_vals)

    def _separate_disconnected_components(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """Separate subtours based on disconnected components in the support graph."""
        threshold = 0.5
        support_edges = [(i, j) for idx, (i, j) in enumerate(self.model.edges) if x_vals[idx] >= threshold]

        if not support_edges:
            return

        n = self.model.n_nodes
        graph = np.zeros((n, n))
        for i, j in support_edges:
            graph[i, j] = 1
            graph[j, i] = 1

        n_components, labels = connected_components(csgraph=csr_matrix(graph), directed=False)
        if n_components <= 1:
            return

        depot_component = labels[self.model.depot]
        for comp_id in range(n_components):
            if comp_id == depot_component:
                continue

            component = set(np.where(labels == comp_id)[0])
            if len(component) < 2:
                continue

            # In VRPP, SEC only applies if at least one node in component is visited
            is_visited = True
            if y_vals is not None:
                is_visited = any(y_vals[node - 1] > 0.5 for node in component if node != self.model.depot)

            if not is_visited:
                continue

            cut_value = self._get_cut_value(component, x_vals)
            violation = 2.0 - cut_value
            if violation > 0.01:
                self.pool.append(PCSubtourEliminationCut(component, violation))

    def _separate_weak_subtours(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """Separate subtours by greedily expanding sets with high internal connectivity."""
        threshold_weak = 0.1
        n = self.model.n_nodes
        graph_weak = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] >= threshold_weak:
                val = x_vals[idx]
                graph_weak[i, j] = val
                graph_weak[j, i] = val

        for center in self.model.customers:
            # Skip if center node not visited
            if y_vals is not None and y_vals[center - 1] < 0.5:
                continue
            node_set = {center}
            candidates = set(self.model.customers) - {center}

            while len(node_set) < min(5, len(self.model.customers)):
                best_node, best_conn = None, 0.0
                for candidate in candidates:
                    conn = sum(graph_weak[candidate, node] for node in node_set)
                    if conn > best_conn:
                        best_conn = conn
                        best_node = candidate

                if best_node is None or best_conn < 0.1:
                    break

                node_set.add(best_node)
                candidates.remove(best_node)

            if len(node_set) >= 2:
                cut_value = self._get_cut_value(node_set, x_vals)
                violation = 2.0 - cut_value
                if violation > 0.01:
                    self.pool.append(PCSubtourEliminationCut(node_set, violation))

    def _get_cut_value(self, node_set: Set[int], x_vals: np.ndarray) -> float:
        """Compute the sum of x values for edges in the cut delta(S)."""
        cut_edges = self.model.delta(node_set)
        return sum(
            x_vals[self.model.edge_to_idx[(min(e), max(e))]]
            for e in cut_edges
            if (min(e), max(e)) in self.model.edge_to_idx
        )

    def _separate_capacity_cuts(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """
        Separate capacity inequalities using demand-based clustering.

        For sets S with high demand, check if enough vehicles (edges crossing cut) exist.
        """
        # Try different seed nodes and grow sets based on demand
        for seed in self.model.customers:
            # Skip if seed node not visited
            if y_vals is not None and y_vals[seed - 1] < 0.5:
                continue
            node_set = {seed}
            remaining = set(self.model.customers) - {seed}
            current_demand = self.model.get_node_demand(seed)

            # Grow set until capacity is reached
            while current_demand < self.model.capacity * 0.9 and remaining:
                # Find nearest node to current set
                best_node = None
                best_dist = float("inf")

                for candidate in remaining:
                    min_dist = min(self.model.cost_matrix[candidate, node] for node in node_set)
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_node = candidate

                if best_node is None:
                    break

                candidate_demand = self.model.get_node_demand(best_node)
                if current_demand + candidate_demand > self.model.capacity * 1.5:
                    break  # Don't grow too large

                node_set.add(best_node)
                remaining.remove(best_node)
                current_demand += candidate_demand

            # Check capacity inequality violation
            if len(node_set) >= 2:
                # In VRPP, we only care about demand of nodes we actually visit
                if y_vals is not None:
                    total_demand = sum(self.model.get_node_demand(i) for i in node_set if i > 0 and y_vals[i - 1] > 0.5)
                else:
                    total_demand = self.model.total_demand(node_set)

                if total_demand <= 0.01:
                    continue

                min_vehicles = int(np.ceil(total_demand / self.model.capacity))

                cut_edges = self.model.delta(node_set)
                cut_value = sum(
                    x_vals[self.model.edge_to_idx[(min(e), max(e))]]
                    for e in cut_edges
                    if (min(e), max(e)) in self.model.edge_to_idx
                )

                required_value = 2.0 * min_vehicles
                violation = required_value - cut_value

                if violation > 0.01:
                    self.pool.append(CapacityCut(node_set, total_demand, self.model.capacity, violation))

    def _separate_capacity_cuts_maxflow_heuristic(  # noqa: C901
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray],
        root_node: bool = False,
        max_cuts: int = 50,
    ):
        """
        Max-Flow-based heuristic separation of Rounded Capacity Cuts (RCCs) for CVRP/VRPP.

        Mathematical Motivation:
        -------------------------
        The Rounded Capacity Cut (RCC) is defined as:
            Σ_{i ∈ S, j ∉ S} x_{ij} ≥ 2 * ⌈d(S) / Q⌉
        where S ⊆ Customers, d(S) is total demand of S, and Q is vehicle capacity.

        Separation Strategy:
        --------------------
        We reformulate the problem as finding a set S such that the cut capacity
        is less than the required fleet size. For a fixed set S, we solve
        max-flow problems between the depot (Node 0) and each customer node i
        in the support graph G*=(V, E*) where edge weights correspond to fractional
        x* values. Following Fischetti et al. (1997), the customer-side of each
        computed min-cut is a candidate set S for an RCC.

        This heuristic method performs extensive separation by solving max-flow to ALL
        nodes with non-zero visitation probability y*_i. While highly rigorous, it operates
        on the fractional support graph and does not strictly guarantee finding all
        mathematically violated facets.

        References:
        -----------
        - Fischetti, M., Salazar-González, J. J., & Toth, P. (1997). "A branch-and-cut
          algorithm for the symmetric generalized traveling salesman problem."
          Operations Research, 45(3), 378-394.
        - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut
          algorithm for the capacitated vehicle routing problem."
          Mathematical Programming, 100(2), 423-445.
        """
        n = self.model.n_nodes
        # Build support graph adjacency
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            val = x_vals[idx]
            if val > 1e-4:
                adj[i, j] = val
                adj[j, i] = val

        # Exact Separation: isolate most fractional nodes to avoid O(N) max-flows.
        # Sort candidates by fractional visitation value descending.
        if y_vals is not None:
            visited_customers = [c for c in self.model.customers if y_vals[c - 1] > 1e-4]
            visited_customers.sort(key=lambda c: y_vals[c - 1], reverse=True)
        else:
            visited_customers = self.model.customers[:]

        # RCIs separate the depot from customers to ensure enough vehicles leave.
        source_node = self.model.depot

        added = 0
        for sink in visited_customers:
            if sink == source_node:
                continue
            if added >= max_cuts:
                break
            try:
                flow_result = maximum_flow(csr_matrix(adj), source_node, sink)
                source_side = self._extract_min_cut(adj, flow_result.flow.toarray(), source_node, sink)
                cut_set = set(range(self.model.n_nodes)) - source_side
            except Exception:
                # Padberg-Rinaldi Min-Cut Fallback
                try:
                    G = nx.Graph()
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] > 1e-4:
                                G.add_edge(u, v, capacity=adj[u, v])
                    cut_value, partition = nx.minimum_cut(G, source_node, sink)
                    s_side, t_side = partition
                    cut_set = t_side if source_node in s_side else s_side
                except Exception:
                    continue

            # Prune invalid cuts: single-node sets, depot in set, or empty sets
            # Single-node sets are already handled by degree constraints (∑ x_ij = 2y_i)
            if not cut_set or len(cut_set) <= 1 or self.model.depot in cut_set:
                continue

            current_y_vals = y_vals if y_vals is not None else np.ones(len(self.model.customers))
            total_demand = sum(self.model.get_node_demand(i) for i in cut_set if i > 0 and current_y_vals[i - 1] > 0.1)

            if total_demand <= 1e-4:
                continue

            min_vehicles = int(np.ceil(total_demand / self.model.capacity))
            if min_vehicles < 1:
                continue

            cut_val = self._get_cut_value(cut_set, x_vals)
            required_val = 2.0 * min_vehicles

            violation = required_val - cut_val
            if violation > 0.01 and not any(
                set(cut_set) == set(existing.node_set) for existing in self.pool if isinstance(existing, CapacityCut)
            ):
                self.pool.append(CapacityCut(set(cut_set), total_demand, self.model.capacity, violation))
                added += 1

    def _separate_pcsec_exact(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray], root_node: bool = False):
        """
        Exact prize-collecting subtour separation using minimum cut (max-flow).
        Fischetti et al. (1997), Section 3.1: GSEC_SEP.

        Finds violated PC-SECs by solving max-flow problems between a reference
        source node s and the depot.
        """
        n = self.model.n_nodes
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            val = x_vals[idx]
            if val > 1e-4:
                adj[i, j] = val
                adj[j, i] = val

        # 1. Source-node selection (s)
        # We select node_i such that y*_i is maximum among nodes i for which y*_i >= eps.
        eps = 0.01
        visited_customers = [c for c in self.model.customers if y_vals is None or y_vals[c - 1] >= eps]
        if not visited_customers:
            return

        source_node = (
            max(visited_customers, key=lambda c: y_vals[c - 1]) if y_vals is not None else visited_customers[0]
        )

        for sink_node in visited_customers:
            if sink_node == source_node or sink_node == self.model.depot:
                continue

            try:
                # Solve max-flow from source_node to sink_node (isolation of S from depot)
                flow_result = maximum_flow(csr_matrix(adj), source_node, sink_node)
                max_flow_value = flow_result.flow_value
                flow_matrix = flow_result.flow.toarray()

                # Extract cut set S reachable from source in residual graph
                s_set = self._extract_min_cut(adj, flow_matrix, source_node, sink_node)
            except Exception:
                # Padberg-Rinaldi Min-Cut Fallback
                try:
                    G = nx.Graph()
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] > 1e-4:
                                G.add_edge(u, v, capacity=adj[u, v])
                    max_flow_value, partition = nx.minimum_cut(G, source_node, sink_node)
                    s_side, t_side = partition
                    s_set = s_side if source_node in s_side else t_side
                except Exception:
                    continue

            if (
                s_set
                and self.model.depot not in s_set
                and len(s_set) >= 2
                and not any(
                    set(s_set) == set(existing.node_set)
                    for existing in self.pool
                    if isinstance(existing, PCSubtourEliminationCut)
                )
            ):
                # Re-select j from N\S to satisfy Form 2.3 requirement j ∉ S.
                not_in_s = set(range(self.model.n_nodes)) - s_set - {self.model.depot}
                if not not_in_s:
                    continue
                if y_vals is not None:
                    actual_j, actual_yj = max(
                        [(v, y_vals[v - 1] if v > 0 else 1.0) for v in not_in_s],
                        key=lambda x: x[1],
                    )
                else:
                    actual_j, actual_yj = next(iter(not_in_s)), 1.0

                yi = y_vals[source_node - 1] if y_vals is not None else 1.0
                actual_rhs = 2.0 * (yi + actual_yj - 1.0)

                # Only add if this recomputed rhs still gives a violated cut.
                if actual_rhs - max_flow_value > 0.01:
                    cut = PCSubtourEliminationCut(
                        set(s_set),
                        actual_rhs - max_flow_value,
                        facet_form="2.3",
                        node_i=source_node,
                        node_j=actual_j,
                    )
                    # RHS = 2*(yi + yj - 1) uses fractional LP values.
                    # This cut is valid for the current LP relaxation but may tighten
                    # below 2*(1+1-1)=2 for integer solutions. Mark as local so it is
                    # removed when backtracking past this B&B node.
                    cut.local_only = True
                    self.pool.append(cut)

    def _extract_min_cut(self, capacity: np.ndarray, flow: np.ndarray, source: int, sink: int) -> Set[int]:
        """
        Extract source-side of minimum cut using BFS on residual graph.
        Paper Section 3.1: the cut set is the set of nodes reachable from
        the source in the residual graph.

        Args:
            capacity: Capacity matrix (support graph edges).
            flow: Flow matrix from max-flow result.
            source: Source node index (depot or reference node).
            sink: Sink node index (customer).

        Returns:
            Set of nodes reachable from source in the residual graph.
        """
        residual = capacity - flow
        visited: Set[int] = set()
        queue = [source]
        visited.add(source)

        while queue:
            u = queue.pop(0)
            for v in range(len(residual)):
                if residual[u, v] > 1e-6 and v not in visited:
                    visited.add(v)
                    queue.append(v)

        return visited

    def _separate_comb_heuristic(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """
        Heuristic comb inequality separation (Theorem 2.3 from Fischetti et al. 1997).

        **NOTE: This method is DISABLED by default (USE_COMB_CUTS = False).**

        Rationale for disabling:
            - Custom comb separation is heuristic and may miss optimal cuts
            - Gurobi's internal polyhedral cuts (Clique, Cover, etc.) are highly optimized
            - Unoptimized Python-level comb separation adds computational overhead
            - For this Natural Edge Formulation, SEC + RCC cuts are sufficient

        A comb inequality is defined by:
        - Handle H: A node subset
        - Teeth T_1, ..., T_s: Node subsets where |T_j ∩ H| = 1 for all j

        The inequality is:
        ∑_{e ∈ E(H)} x_e + ∑_{j=1}^{s} ∑_{e ∈ E(T_j)} x_e ≤ |H| + ∑_{j} |T_j| - (s+1)/2

        This implementation:
        1. Identifies candidate handles from high-degree nodes in support graph
        2. Grows handles using greedy BFS
        3. Finds teeth that have exactly one node in the handle
        4. Evaluates violation and adds to pool
        """
        # In VRPP, comb inequalities are less common and tricky with node selection
        # For now, skip if any node in support graph for this handle/teeth is unvisited
        # This is a simplification.
        # Build support graph with edges that have significant fractional values
        threshold = 0.3
        support_edges = []
        edge_weights: Dict[Tuple[int, int], float] = {}

        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] >= threshold:
                support_edges.append((i, j))
                edge_weights[(i, j)] = x_vals[idx]
                edge_weights[(j, i)] = x_vals[idx]

        if len(support_edges) < 3:
            return

        # Build adjacency list from support edges
        adjacency: Dict[int, List[int]] = {i: [] for i in range(self.model.n_nodes)}
        for i, j in support_edges:
            adjacency[i].append(j)
            adjacency[j].append(i)

        # Find candidate handles: nodes with high degree in support graph
        high_degree_nodes = []
        for node in range(self.model.n_nodes):
            if len(adjacency[node]) >= 3:
                high_degree_nodes.append((node, len(adjacency[node])))

        # Sort by degree descending
        high_degree_nodes.sort(key=lambda x: x[1], reverse=True)

        # Try multiple handle seeds
        max_attempts = min(10, len(high_degree_nodes))
        for attempt in range(max_attempts):
            if attempt >= len(high_degree_nodes):
                break

            seed_node = high_degree_nodes[attempt][0]

            # Grow handle using BFS from seed
            handle = self._grow_handle(seed_node, adjacency, edge_weights, max_size=15)

            if len(handle) < 3:
                continue

            # Find teeth for this handle
            teeth = self._find_teeth_for_handle(handle, adjacency, edge_weights)

            if len(teeth) < 3 or len(teeth) % 2 == 0:
                # Need odd number of teeth >= 3 for valid comb
                continue

            # Compute violation
            violation = self._compute_comb_violation(handle, teeth, x_vals)

            if violation > 0.01:
                self.pool.append(CombInequality(handle, teeth, violation))

    def _grow_handle(
        self,
        seed: int,
        adjacency: Dict[int, List[int]],
        edge_weights: Dict[Tuple[int, int], float],
        max_size: int = 15,
    ) -> Set[int]:
        """
        Grow a handle from a seed node using greedy BFS.

        Args:
            seed: Starting node for handle.
            adjacency: Adjacency list from support graph.
            edge_weights: Edge weights (fractional x values).
            max_size: Maximum handle size.

        Returns:
            Set of nodes forming the handle.
        """
        handle = {seed}
        candidates = set(adjacency[seed])

        while len(handle) < max_size and candidates:
            # Find candidate with strongest connection to handle
            best_node = None
            best_connection_strength = -1.0

            for candidate in candidates:
                if candidate in handle:
                    continue

                # Sum of edge weights connecting candidate to handle
                connection_strength = sum(
                    edge_weights.get((candidate, h), 0.0) for h in handle if (candidate, h) in edge_weights
                )

                if connection_strength > best_connection_strength:
                    best_connection_strength = connection_strength
                    best_node = candidate

            if best_node is None or best_connection_strength < 0.2:
                break

            # Add to handle
            handle.add(best_node)

            # Update candidates
            for neighbor in adjacency[best_node]:
                if neighbor not in handle:
                    candidates.add(neighbor)

        return handle

    def _find_teeth_for_handle(
        self,
        handle: Set[int],
        adjacency: Dict[int, List[int]],
        edge_weights: Dict[Tuple[int, int], float],
    ) -> List[Set[int]]:
        """
        Find teeth for a given handle.

        A tooth T_j must satisfy:
        - |T_j ∩ H| = 1 (exactly one node in handle)
        - |T_j| >= 3 (at least 3 nodes total)
        - T_j forms a connected component

        Args:
            handle: Handle node set.
            adjacency: Adjacency list.
            edge_weights: Edge weights.

        Returns:
            List of teeth (each tooth is a set of nodes).
        """
        teeth: List[Set[int]] = []

        # For each node in the handle, try to grow a tooth from it
        for anchor in handle:
            tooth = self._grow_tooth(anchor, handle, adjacency, edge_weights)

            if tooth is None:
                continue

            # Validate tooth
            intersection = tooth & handle
            if len(intersection) != 1:
                continue

            if len(tooth) < 3:
                continue

            # Check it's not a duplicate
            is_duplicate = any(tooth == existing_tooth for existing_tooth in teeth)
            if is_duplicate:
                continue

            teeth.append(tooth)

            # Limit number of teeth
            if len(teeth) >= 7:
                break

        return teeth

    def _grow_tooth(
        self,
        anchor: int,
        handle: Set[int],
        adjacency: Dict[int, List[int]],
        edge_weights: Dict[Tuple[int, int], float],
        max_size: int = 7,
    ) -> Optional[Set[int]]:
        """
        Grow a tooth from an anchor node in the handle.

        The tooth starts with the anchor (which is in handle) and grows
        by adding nodes outside the handle that are well-connected.

        Args:
            anchor: Node in handle to anchor the tooth.
            handle: Handle node set.
            adjacency: Adjacency list.
            edge_weights: Edge weights.
            max_size: Maximum tooth size.

        Returns:
            Tooth node set, or None if no valid tooth found.
        """
        # Start tooth with the anchor
        tooth = {anchor}

        # Find neighbors of anchor that are NOT in handle
        external_neighbors = [neighbor for neighbor in adjacency[anchor] if neighbor not in handle]

        if not external_neighbors:
            return None

        # Greedily add nodes outside the handle
        candidates = set(external_neighbors)

        while len(tooth) < max_size and candidates:
            # Find best candidate to add
            best_node = None
            best_connection = -1.0

            for candidate in candidates:
                if candidate in handle or candidate in tooth:
                    continue

                # Connection strength to current tooth nodes (excluding anchor)
                connection = sum(edge_weights.get((candidate, t), 0.0) for t in tooth if (candidate, t) in edge_weights)

                if connection > best_connection:
                    best_connection = connection
                    best_node = candidate

            if best_node is None or best_connection < 0.15:
                break

            tooth.add(best_node)

            # Update candidates
            for neighbor in adjacency[best_node]:
                if neighbor not in handle and neighbor not in tooth:
                    candidates.add(neighbor)

        # Valid tooth must have at least 3 nodes
        if len(tooth) < 3:
            return None

        return tooth

    def _compute_comb_violation(
        self,
        handle: Set[int],
        teeth: List[Set[int]],
        x_vals: np.ndarray,
    ) -> float:
        """
        Compute the degree of violation for a comb inequality.

        Comb inequality:
        ∑_{e ∈ E(H)} x_e + ∑_{j=1}^{s} ∑_{e ∈ E(T_j)} x_e ≤ |H| + ∑_{j} |T_j| - (s+1)/2

        Args:
            handle: Handle node set.
            teeth: List of teeth.
            x_vals: Fractional edge values.

        Returns:
            Degree of violation (LHS - RHS).
        """
        # Compute LHS: sum of edges in handle and teeth
        lhs = 0.0

        # Edges in handle
        handle_edges = self.model.edges_in_set(handle)
        for edge in handle_edges:
            edge_idx = self.model.edge_to_idx.get(tuple(sorted(edge)))
            if edge_idx is not None:
                lhs += x_vals[edge_idx]

        # Edges in teeth
        for tooth in teeth:
            tooth_edges = self.model.edges_in_set(tooth)
            for edge in tooth_edges:
                edge_idx = self.model.edge_to_idx.get(tuple(sorted(edge)))
                if edge_idx is not None:
                    lhs += x_vals[edge_idx]

        # Compute RHS
        s = len(teeth)
        rhs = len(handle) + sum(len(t) for t in teeth) - (s + 1) / 2.0

        # Violation
        violation = lhs - rhs

        return violation

    def _separate_gsec_h2(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """
        Kruskal-based Heuristic (GSEC_H2) for GSEC separation.

        Algorithm:
        1. Sort edges e with x_e* > 0 in descending order.
        2. Construct forest using Kruskal's merging logic.
        3. For each merge (Va, Vb), evaluate GSEC violation for S = Va U Vb.
        4. GSEC: sum_{e in delta(S)} x_e >= 2(y_i + y_j - 1)
           where i in S and j not in S have highest y* values.
        """
        # Step 1: Edge Sorting
        support_edges = []
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] > 1e-4:
                support_edges.append((i, j, x_vals[idx]))

        support_edges.sort(key=lambda x: x[2], reverse=True)

        # Step 2: Forest Construction (DSU)
        n = self.model.n_nodes
        parent = list(range(n))
        components: Dict[int, Set[int]] = {i: {i} for i in range(n)}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        y_vals_full = np.ones(n)
        if y_vals is not None:
            y_vals_full[1:] = y_vals

        for u, v, _ in support_edges:
            root_u = find(u)
            root_v = find(v)

            if root_u != root_v:
                # Potential Merge: Check violation for S = Va U Vb
                # Merge
                new_component = components[root_u] | components[root_v]
                parent[root_u] = root_v
                components[root_v] = new_component
                del components[root_u]

                # Step 3: Violation Check for S = new_component
                if self.model.depot in new_component:
                    continue  # S must not contain depot for SEC/GSEC cut

                if len(new_component) < 2:
                    continue

                # Find i in S with max y*
                max_y_in = max(y_vals_full[node] for node in new_component)
                # Find j not in S with max y*
                not_in_s = set(range(n)) - new_component
                max_y_out = max(y_vals_full[node] for node in not_in_s)

                gsec_rhs = 2.0 * (max_y_in + max_y_out - 1.0)
                if gsec_rhs <= 0.0:
                    continue

                cut_val = self._get_cut_value(new_component, x_vals)
                violation = gsec_rhs - cut_val

                if violation > 0.01:
                    self.pool.append(PCSubtourEliminationCut(set(new_component), violation))

    def _strengthen_pool(self, ineq_list: List[Inequality], x_vals: np.ndarray, y_vals: Optional[np.ndarray]):
        """
        Apply Procedure PCSEC_BUILD to all subtour sets.
        Refines sets S and selects the strongest facet form (2.1, 2.2, 2.3).
        """
        for i in range(len(ineq_list)):
            cut = ineq_list[i]
            if not isinstance(cut, PCSubtourEliminationCut):
                continue

            # 1. Refine set S using PCSEC_BUILD logic
            s_set = self._refine_pcsec_build(cut.node_set, x_vals, y_vals)
            cut.node_set = s_set

            # 2. Select strongest facet form
            # Identify i in S and j not in S with max y*
            max_yi = 1.0
            best_i = next(iter(s_set))
            if y_vals is not None:
                y_in = [(node, y_vals[node - 1]) for node in s_set if node > 0]
                if y_in:
                    best_i, max_yi = max(y_in, key=lambda x: x[1])

            max_yj = 1.0
            best_j = self.model.depot
            if y_vals is not None:
                not_s = set(range(self.model.n_nodes)) - s_set
                y_out = [(node, y_vals[node - 1] if node > 0 else 1.0) for node in not_s]
                if y_out:
                    best_j, max_yj = max(y_out, key=lambda x: x[1])

            # Recalculate violation for each form (2.1, 2.2, 2.3)
            cut_val = self._get_cut_value(s_set, x_vals)

            # Form 2.3: sum x_e >= 2(yi + yj - 1)
            vio_2_3 = (2.0 * (max_yi + max_yj - 1.0) - cut_val) if y_vals is not None else (2.0 - cut_val)

            # Form 2.2: sum x_e >= 2yi
            vio_2_2 = (2.0 * max_yi - cut_val) if y_vals is not None else (2.0 - cut_val)

            # Form 2.1: sum x_e >= 2
            vio_2_1 = 2.0 - cut_val

            # Select strongest (max violation)
            if vio_2_1 > vio_2_2 + 1e-4 and vio_2_1 > vio_2_3 + 1e-4:
                cut.facet_form = "2.1"
                cut.rhs = 2.0
                cut.violation = vio_2_1
                cut.local_only = False  # Form 2.1 is always global
            # Form 2.2: Σ x_e - 2y_i >= 0
            elif vio_2_2 > vio_2_3 + 1e-4:
                cut.facet_form = "2.2"
                cut.node_i = best_i
                cut.rhs = 0.0
                cut.violation = vio_2_2
                cut.local_only = True  # Depends on specific y_i, mark local
            # Form 2.3: Σ x_e - 2y_i - 2y_j >= -2
            else:
                cut.facet_form = "2.3"
                cut.node_i = best_i
                cut.node_j = best_j
                cut.rhs = -2.0
                cut.violation = vio_2_3
                cut.local_only = True  # Depends on y_i, y_j, mark local

    def _refine_pcsec_build(self, s_set: Set[int], x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> Set[int]:
        """
        Procedure PCSEC_BUILD (Fischetti et al. 1997, Section 3.1).
        Attempts to refine the set S by moving nodes to minimize cut value sum_{e in delta(S)} x_e.
        """
        refined_s = set(s_set)
        improved = True

        while improved:
            improved = False
            for u in list(refined_s):
                # Delta cut value if u is moved from S to V\S
                # delta_val = edges(u, V\S) - edges(u, S)
                in_s_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in refined_s if u != v)
                all_u_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in range(self.model.n_nodes) if v != u)
                out_s_sum = all_u_sum - in_s_sum

                # We want to minimize sum_{e in delta(S)} x_e.
                # Moving u out changes it from out_s_sum to in_s_sum inside the cut.
                # Actually, sum_{e in delta(S)} = sum_{i in S, j not in S} x_ij.
                # Moving u from S to V\S removes sum_{j in V\S} x_uj and adds sum_{j in S} x_uj.
                if in_s_sum - out_s_sum < -0.01 and len(refined_s) > 2:
                    refined_s.remove(u)
                    improved = True
                    break

            if improved:
                continue

            for u in range(self.model.n_nodes):
                if u in refined_s or u == self.model.depot:
                    continue

                # Check moving u from V\S into S
                in_s_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in refined_s)
                all_u_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in range(self.model.n_nodes) if v != u)
                out_s_sum = all_u_sum - in_s_sum

                if out_s_sum - in_s_sum < -0.01 and u != self.model.depot:
                    # Adding u to S reduces cut capacity
                    refined_s.add(u)
                    improved = True
                    break

        return refined_s
