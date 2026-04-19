"""
Branch-and-Price Solver for VRPP.

Implements the complete Branch-and-Price algorithm with:
- Column generation at each node
- Pluggable branching strategy (edge branching or Ryan-Foster)
- LP relaxation solving with pricing
- Integer solution recovery

Based on Sections 5 and 6 of Barnhart et al. (1998).

Reference:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P., & Vance, P. H. (1998).
    "Branch-and-price: Column Generation for Solving Huge Integer Programs".
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    BranchNode,
    RCSPPSolver,
    Route,
    VRPPMasterProblem,
)

from .params import BPParams
from .pricing_subproblem import PricingSubproblem


class BranchAndPriceSolver:
    """
    Branch-and-Price Algorithm for VRPP with Column Generation.

    Algorithm:
        1. Initialise with greedy routes.
        2. Solve LP relaxation with column generation.
        3. If LP solution is integer, terminate.
        4. Otherwise branch using the configured strategy and recurse.

    Branching strategies
    --------------------
    ``"edge"`` (default)
        Branches on the most-fractional directed arc (x_{uv}).  Integrates
        natively with the DP label extension.

    ``"ryan_foster"``
        Selection of a node-pair (r, s) to branch on. Produces two
        child nodes where r and s MUST or MUST NOT appear in the same
        route. Note: This is theoretically a heuristic when using a
        Set Covering master problem (default).
        **WARNING:** Ryan-Foster branching loses its theoretical exactness
        guarantee when applied to a Set Covering master problem (>= 1), as it
        can erroneously prune optimal over-covering solutions. Use 'edge'
        branching for rigorous proofs of optimality.

    Label-Correcting Algorithm
    --------------------------
    When ``use_exact_pricing=True``, the DP-based ``RCSPPSolver`` is used.
    This employs a **Label-Correcting** algorithm (FIFO queue) to handle
    potential negative edge costs during column generation. Two sub-modes
    are available via ``use_ng_routes``:

    ``use_ng_routes=True`` (default)
        Uses ng-route relaxation (Baldacci et al. 2011) — far fewer labels,
        scalable to large instances.

    ``use_ng_routes=False``
        Falls back to exact ESPPRC — every generated route is elementary,
        but the label count grows exponentially with instance size.
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
        max_iterations: int = 100,
        max_routes_per_iteration: int = 10,
        optimality_gap: float = 1e-4,
        use_ryan_foster: bool = False,
        branching_strategy: str = "edge",
        tree_search_strategy: str = "best_first",
        max_branch_nodes: int = 1000,
        use_exact_pricing: bool = False,
        vehicle_limit: Optional[int] = None,
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
        cleanup_frequency: int = 20,
        cleanup_threshold: float = -100.0,
        early_termination_gap: float = 1e-3,
        multiple_waste_types: bool = False,
        allow_heuristic_ryan_foster: bool = False,
        params: Optional[BPParams] = None,
    ) -> None:
        """
        Initialise the Branch-and-Price solver.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = 0).
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
            wastes: Mapping from node ID to waste volume.
            capacity: Vehicle payload capacity.
            revenue_per_kg: Revenue per unit of waste collected.
            cost_per_km: Operating cost per unit of distance travelled.
            mandatory_nodes: Node indices that must be visited.
            params: Standardized BP parameters.
        """
        if params is None:
            # Create a params object from explicit arguments for backward compatibility
            params = BPParams(
                max_iterations=max_iterations,
                max_routes_per_iteration=max_routes_per_iteration,
                optimality_gap=optimality_gap,
                branching_strategy=branching_strategy,
                max_branch_nodes=max_branch_nodes,
                use_exact_pricing=use_exact_pricing,
                vehicle_limit=vehicle_limit,
                use_ng_routes=use_ng_routes,
                ng_neighborhood_size=ng_neighborhood_size,
                cleanup_frequency=cleanup_frequency,
                cleanup_threshold=cleanup_threshold,
                early_termination_gap=early_termination_gap,
                multiple_waste_types=multiple_waste_types,
                allow_heuristic_ryan_foster=allow_heuristic_ryan_foster,
                tree_search_strategy=tree_search_strategy,
            )
            # Handle legacy Ryan-Foster promotion
            if use_ryan_foster and params.branching_strategy == "edge":
                params.branching_strategy = "ryan_foster"

        self.params = params
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.depot = 0

        # Tracking exactness
        self.proven_optimal: bool = True

        # Precompute ng-neighborhoods once for reuse across all B&B nodes.
        self._ng_neighborhoods: Optional[Dict[int, Set[int]]] = None
        if self.params.use_exact_pricing:
            from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver as _R

            _tmp = _R(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                use_ng_routes=self.params.use_ng_routes,
                ng_neighborhood_size=self.params.ng_neighborhood_size,
            )
            self._ng_neighborhoods = _tmp.ng_neighborhoods

        # Resolve branching strategy using central logic.
        branching_strategy = self._resolve_branching_strategy(
            branching_strategy=branching_strategy,
            use_ryan_foster=use_ryan_foster,
            values={},  # Values handled at policy level, but solver respects init args
        )

        if branching_strategy == "divergence" and not self.params.multiple_waste_types:
            logging.info(
                "Single-commodity VRP detected. Promoting 'divergence' branching to 'multi_edge_partition' strategy."
            )
            branching_strategy = "multi_edge_partition"

        self.params.branching_strategy = branching_strategy
        self.params.tree_search_strategy = tree_search_strategy
        self.params.use_ryan_foster = branching_strategy == "ryan_foster"

        if self.params.use_ryan_foster and not self.params.allow_heuristic_ryan_foster:
            raise ValueError(
                "Ryan-Foster branching is theoretically inexact when used with a Set Covering "
                "master problem (the default). It can prune optimal over-covering solutions. "
                "To acknowledge this and proceed, pass 'allow_heuristic_ryan_foster=True' to the solver, "
                "or use the mathematically exact 'edge' branching strategy."
            )

        if self.params.use_ryan_foster:
            warnings.warn(
                "Ryan-Foster branching is theoretically a heuristic when used with a Set Covering "
                "master problem (>= 1). It may lead to unbalanced trees or over-coverage issues. "
                "Edge branching is the mathematically exact default.",
                UserWarning,
                stacklevel=2,
            )

        # Statistics
        self.num_iterations: int = 0
        self.num_columns_generated: int = 0
        self.lp_bound: float = 0.0
        self.ip_solution: float = 0.0
        self.tree: Optional[BranchAndBoundTree] = None

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[int], float, Dict[str, Any]]:
        # Use the full B&P tree for all branching strategies.
        # _solve_without_branching is kept as a fast path only when
        # branching_strategy is explicitly set to "none" or similar sentinel.
        if self.params.branching_strategy in ("edge", "ryan_foster", "divergence", "multi_edge_partition"):
            return self._solve_with_branching()  # type: ignore[return-value]
        return self._solve_without_branching()

    # ------------------------------------------------------------------
    # Solve without branching (column generation + direct IP)
    # ------------------------------------------------------------------

    def _solve_without_branching(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """Solve with column generation followed by a direct MIP solve."""
        master = self._make_master()
        heuristic_pricing, exact_pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(heuristic_pricing)
        master.build_model(initial_routes)

        lp_obj, route_values = self._column_generation(master, (heuristic_pricing, exact_pricing))
        self.lp_bound = lp_obj

        if self._is_integer_solution(route_values):
            selected = [master.routes[i] for i, v in route_values.items() if v > 0.5]
            return self._routes_to_tour(selected), lp_obj, self._get_statistics()

        ip_obj, selected = master.solve_ip()
        self.ip_solution = ip_obj
        return self._routes_to_tour(selected), ip_obj, self._get_statistics()

    # ------------------------------------------------------------------
    # Solve with branching
    # ------------------------------------------------------------------

    def _solve_with_branching(self) -> Tuple[List[int], Optional[float], Dict[str, Any]]:
        """Solve using the configured branching strategy in a B&B tree."""
        self.tree = BranchAndBoundTree(
            v_model=None,
            strategy=self.params.branching_strategy,
            search_strategy=self.params.tree_search_strategy,
        )

        root_lp, root_values, root_routes = self._solve_node(self.tree.root)
        self.tree.root.route_values = root_values
        self.tree.root.routes = root_routes
        self.tree.root.is_integer = self._is_integer_solution(root_values)
        self.lp_bound = self.tree.root.lp_bound  # type: ignore[assignment]

        if self.tree.root.is_integer:
            selected = [root_routes[i] for i, v in root_values.items() if v > 0.5]
            return self._routes_to_tour(selected), root_lp, self._get_statistics()

        self.tree.nodes_explored += 1

        # Remove root from open list — it has already been evaluated above.
        if self.tree.root in self.tree.open_nodes:
            self.tree.open_nodes.remove(self.tree.root)

        routes: List[Route] = root_routes
        while not self.tree.is_empty() and self.tree.nodes_explored < self.params.max_branch_nodes:
            node = self.tree.get_next_node()
            if node is None:
                break

            if (
                self.tree.best_integer_solution is not None
                and node.lp_bound is not None
                and node.lp_bound <= self.tree.best_integer_solution
            ):
                self.tree.nodes_pruned += 1
                continue

            lp_obj, route_values, routes = self._solve_node(node, parent_routes=routes)
            node.route_values = route_values
            node.routes = routes

            if lp_obj is None:
                node.is_infeasible = True
                self.tree.nodes_pruned += 1
                continue

            if self._is_integer_solution(route_values):
                node.is_integer = True
                node.ip_solution = lp_obj
                if self.tree.update_incumbent(node, lp_obj):
                    self.ip_solution = lp_obj
                    self.tree.prune_by_bound()
                continue

            result = self.tree.branch(node, routes, route_values, mandatory_nodes=self.mandatory_nodes)
            if result is None:
                logging.warning(
                    "Branching stalled (no fractional candidates found). Falling back to restricted "
                    "MIP solve. This is a Primal Heuristic; global optimality is no longer guaranteed."
                )
                self.proven_optimal = False
                master = self._build_master_for_node(node, routes)
                ip_obj, selected = master.solve_ip()
                if ip_obj is not None:
                    node.is_integer = True
                    node.ip_solution = ip_obj
                    node.routes = routes
                    node.route_values = {i: 1.0 if r in selected else 0.0 for i, r in enumerate(routes)}
                    if self.tree.update_incumbent(node, ip_obj):
                        self.ip_solution = ip_obj
                        self.tree.prune_by_bound()
            else:
                left_child, right_child = result
                self.tree.add_node(left_child)
                self.tree.add_node(right_child)

        if self.tree.best_integer_node is not None:
            rv = self.tree.best_integer_node.route_values
            node_routes = self.tree.best_integer_node.routes
            if rv and node_routes:
                selected = [node_routes[i] for i, v in rv.items() if v > 0.5 and i < len(node_routes)]
                return (
                    self._routes_to_tour(selected),
                    self.tree.best_integer_solution,  # type: ignore[return-value]
                    self._get_statistics(),
                )

        root_master = self._build_master_for_node(self.tree.root, root_routes)
        ip_obj, selected = root_master.solve_ip()
        if ip_obj is not None:
            self.ip_solution = ip_obj
            return self._routes_to_tour(selected), ip_obj, self._get_statistics()

        return [0], 0.0, self._get_statistics()

    # ------------------------------------------------------------------
    # Node solving
    # ------------------------------------------------------------------

    def _solve_node(
        self,
        node: BranchNode,
        parent_routes: Optional[List[Route]] = None,
    ) -> Tuple[Optional[float], Dict[int, float], List[Route]]:
        """Solve the LP relaxation at a single B&B node via column generation."""
        master = self._make_master()
        constraints = node.get_all_constraints()
        heuristic_pricing, exact_pricing = self._make_pricing()

        # Use parent column pool (filtered to node constraints) when available.
        # Fall back to greedy initial routes only if no feasible parent columns exist.
        feasible = [r for r in parent_routes if node.is_route_feasible(r)] if parent_routes else []

        if not feasible:
            initial_routes = self._generate_initial_routes(heuristic_pricing)
            feasible = [r for r in initial_routes if node.is_route_feasible(r)]

        master.build_model(feasible)

        try:
            lp_obj, route_values, best_bound = self._column_generation_with_constraints(
                master, (heuristic_pricing, exact_pricing), node, constraints
            )
            node.lp_bound = min(lp_obj, best_bound)

            # Part 5 Fix 5: Rounded Capacity Cut (RCC) separation at root node
            # We only separate at depth 0 to avoid cut management complexity in the B&B tree.
            if node.depth == 0:
                logging.info("Starting Root RCC separation...")
                cuts_added = master.find_and_add_violated_rcc(route_values, master.routes)
                if cuts_added > 0:
                    logging.info(f"Added {cuts_added} capacity cuts at root. Re-solving...")
                    lp_obj, route_values, best_bound = self._column_generation_with_constraints(
                        master, (heuristic_pricing, exact_pricing), node, constraints
                    )
                    node.lp_bound = min(lp_obj, best_bound)
        except RuntimeError:
            return None, {}, []

        return lp_obj, route_values, master.routes

    # ------------------------------------------------------------------
    # Column generation
    # ------------------------------------------------------------------

    def _column_generation(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
    ) -> Tuple[float, Dict[int, float]]:
        """Perform unconstrained column generation with heuristic fallback."""
        heuristic_pricer, exact_pricer = pricing
        last_basis: Optional[Tuple[List[int], List[int]]] = None

        for i in range(self.params.max_iterations):
            self.num_iterations += 1

            if last_basis is not None:
                master.restore_basis(*last_basis)

            lp_obj, route_values = master.solve_lp_relaxation()
            last_basis = master.save_basis()

            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=None)  # type: ignore[arg-type]

            # 2. Fallback to exact pricer if heuristic fails
            if (
                not new_routes or max(rc for _, rc in new_routes) < self.params.optimality_gap
            ) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=None)  # type: ignore[arg-type]

            if not new_routes:
                break

            max_rc = max(rc for _, rc in new_routes)
            if max_rc < self.params.optimality_gap:
                break

            # Estimate fleet limit for unconstrained instances to calculate Lagrangian bound
            limit = self.params.vehicle_limit if self.params.vehicle_limit is not None else self.n_nodes
            lagrangian_bound = lp_obj + max_rc * limit
            if abs(lp_obj - lagrangian_bound) < self.params.early_termination_gap:
                break

            self._add_routes_to_master(master, heuristic_pricer, new_routes)
            last_basis = None  # basis is stale after column additions; force fresh solve

            # Task 3: Column Pool Management
            if (i + 1) % self.params.cleanup_frequency == 0:
                master.purge_useless_columns(self.params.cleanup_threshold)

        if last_basis is not None:
            master.restore_basis(*last_basis)
        return master.solve_lp_relaxation()

    def _column_generation_with_constraints(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
        node: BranchNode,
        constraints: List[AnyBranchingConstraint],
    ) -> Tuple[float, Dict[int, float], float]:
        """Perform constrained column generation with heuristic fallback."""
        heuristic_pricer, exact_pricer = pricing
        best_lagrangian = float("inf")
        last_basis: Optional[Tuple[List[int], List[int]]] = None

        for i in range(self.params.max_iterations):
            self.num_iterations += 1
            if last_basis is not None:
                master.restore_basis(*last_basis)

            lp_obj, route_values = master.solve_lp_relaxation()
            last_basis = master.save_basis()
            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]

            # 2. Fallback to exact pricer if heuristic fails
            if (
                not new_routes or max(rc for _, rc in new_routes) < self.params.optimality_gap
            ) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]

            if not new_routes:
                break

            feasible: List[Tuple[Route, float]] = []
            for route_nodes, rc in new_routes:
                cost, revenue, load, cov = heuristic_pricer.compute_route_details(route_nodes)
                route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
                # Exact solver already respects constraints via state enforcement.
                # Heuristic pricer requires post-hoc check for non-edge constraints (like Ryan-Foster).
                if (self.params.use_exact_pricing and exact_pricer is not None) or node.is_route_feasible(route):
                    feasible.append((route, rc))

            if not feasible:
                break

            # Task 4 & Part 5 Task 3: LP Termination via Lagrangian Bound
            max_rc = max(rc for _, rc in feasible)
            if max_rc < self.params.optimality_gap:
                break

            limit = self.params.vehicle_limit if self.params.vehicle_limit is not None else self.n_nodes
            lagrangian_bound = lp_obj + max_rc * limit
            best_lagrangian = min(best_lagrangian, lagrangian_bound)

            # Task 4: Basis Restoration
            # Basis is reset to None if the variable count changes significantly,
            # but Gurobi handles basis extension efficiently after add_route.
            # We explicitly save AFTER solving and Restore BEFORE solving.
            if abs(lp_obj - lagrangian_bound) < self.params.early_termination_gap:
                break

            for route, _ in feasible:
                master.add_route(route)
            self.num_columns_generated += len(feasible)

            # Task 3: Column Pool Management
            if (i + 1) % self.params.cleanup_frequency == 0:
                master.purge_useless_columns(self.params.cleanup_threshold)

        # Final solve for clean duals/solution
        if last_basis is not None:
            master.restore_basis(*last_basis)
        lp_obj, route_values = master.solve_lp_relaxation()
        return lp_obj, route_values, best_lagrangian

    # ------------------------------------------------------------------
    # Pricing dispatch
    # ------------------------------------------------------------------

    def _call_pricing(
        self,
        pricing: Union[PricingSubproblem, RCSPPSolver],
        dual_values: Dict[Union[int, frozenset[int], str, Tuple[int, int]], float],
        constraints: Optional[List[AnyBranchingConstraint]],
    ) -> List[Tuple[List[int], float]]:
        """
        Dispatch a pricing call to the correct solver with the right kwargs.

        RCSPPSolver uses ``branching_constraints``; PricingSubproblem uses
        ``active_constraints``.
        """
        from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver

        if isinstance(pricing, RCSPPSolver):
            routes = pricing.solve(
                dual_values=cast(Any, dual_values),
                max_routes=self.params.max_routes_per_iteration,
                branching_constraints=constraints,
            )
            # RCSPPSolver now returns Route objects; convert to (path, reduced_cost) for BP backward compatibility
            return [(r.nodes, r.reduced_cost if r.reduced_cost is not None else 0.0) for r in routes]
        else:
            return pricing.solve(
                dual_values=cast(Any, dual_values),
                max_routes=self.params.max_routes_per_iteration,
                active_constraints=constraints,
            )

    # ------------------------------------------------------------------
    # Initial column generation
    # ------------------------------------------------------------------

    def _generate_initial_routes(self, pricing: Any) -> List[Route]:
        """
        Generate high-quality initial feasible columns for the master problem.

        Uses a Greedy Nearest Neighbor heuristic to cover all mandatory nodes
        with feasible routes, obeying capacity constraints. This provides a
        stronger initial LP bound and more stable early dual variables than
        simple artificial out-and-back routes.

        Args:
            pricing: The pricing subproblem (used to compute cost/revenue).

        Returns:
            List of feasible Route objects covering all mandatory nodes.
        """
        routes: List[Route] = []
        uncovered = sorted(list(self.mandatory_nodes))

        while uncovered:
            current_route: List[int] = []
            current_load = 0.0
            current_node = self.depot

            while True:
                # Find nearest uncovered mandatory node that fits.
                best_next = None
                best_dist = float("inf")

                for candidate in uncovered:
                    demand = self.wastes.get(candidate, 0.0)
                    if current_load + demand <= self.capacity:
                        dist = self.cost_matrix[current_node, candidate]
                        if dist < best_dist:
                            best_dist = dist
                            best_next = candidate

                if best_next is None:
                    break  # Capacity reached or no closer nodes.

                current_route.append(best_next)
                current_load += self.wastes.get(best_next, 0.0)
                uncovered.remove(best_next)
                current_node = best_next

            # Finalize route return to depot.
            if current_route:
                cost, revenue, load, coverage = pricing.compute_route_details(current_route)
                routes.append(
                    Route(
                        nodes=current_route,
                        cost=cost,
                        revenue=revenue,
                        load=load,
                        node_coverage=coverage,
                    )
                )

        return routes

    def _add_routes_to_master(
        self,
        master: VRPPMasterProblem,
        pricing: Any,
        new_routes: List[Tuple[List[int], float]],
    ) -> None:
        """Add a batch of new route candidates to the master problem."""
        for route_nodes, _ in new_routes:
            cost, revenue, load, cov = pricing.compute_route_details(route_nodes)
            route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
            master.add_route(route)
            self.num_columns_generated += 1

    # ------------------------------------------------------------------
    # Tour construction with deduplication
    # ------------------------------------------------------------------

    def _routes_to_tour(self, routes: List[Route]) -> List[int]:
        """
        Convert selected routes to a flat tour, deduplicating customer visits.

        Because the master problem uses Set Covering (≥ 1), the IP may select
        overlapping routes.  Each customer node is included exactly once in the
        output; subsequent routes that would revisit a node simply skip it.
        Empty routes (all nodes already visited) are omitted to avoid bare
        depot → depot arcs.

        Returns:
            Flat tour: [depot, n1, n2, ..., depot, n3, ..., depot].
        """
        tour: List[int] = [self.depot]
        visited_nodes: Set[int] = {self.depot}

        for route in routes:
            unique_nodes = [n for n in route.nodes if n not in visited_nodes]
            if not unique_nodes:
                continue
            tour.extend(unique_nodes)
            tour.append(self.depot)
            visited_nodes.update(unique_nodes)

        return tour

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_master(self) -> VRPPMasterProblem:
        """Construct a fresh VRPPMasterProblem with the solver's parameters."""
        return VRPPMasterProblem(
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            vehicle_limit=self.params.vehicle_limit,
        )

    def _make_pricing(self) -> Tuple[PricingSubproblem, Optional[RCSPPSolver]]:
        """
        Construct the pricing solvers for this configuration.

        Returns:
            Tuple of (heuristic_pricer, exact_pricer).
            heuristic_pricer: Always provided (PricingSubproblem).
            exact_pricer: RCSPPSolver if use_exact_pricing=True, else None.
        """
        kwargs: Dict[str, Any] = dict(
            n_nodes=self.n_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

        heuristic_pricer = PricingSubproblem(**kwargs)
        exact_pricer = None

        if self.params.use_exact_pricing:
            exact_pricer = RCSPPSolver(
                **kwargs,
                use_ng_routes=self.params.use_ng_routes,
                ng_neighborhood_size=self.params.ng_neighborhood_size,
                ng_neighborhoods=self._ng_neighborhoods,
            )

        return heuristic_pricer, exact_pricer

    def _build_master_for_node(self, node: BranchNode, routes: List[Route]) -> VRPPMasterProblem:
        """Build a master problem pre-loaded with constraint-feasible routes."""
        master = self._make_master()
        feasible = [r for r in routes if node.is_route_feasible(r)]
        master.build_model(feasible)
        return master

    def _is_integer_solution(self, route_values: Dict[int, float], tol: float = 1e-5) -> bool:
        """Return True if every LP route value is within *tol* of 0 or 1."""
        return all(abs(v - round(v)) <= tol for v in route_values.values())

    @staticmethod
    def _resolve_branching_strategy(
        branching_strategy: str,
        use_ryan_foster: bool,
        values: Dict[str, Any],
    ) -> str:
        """Single source of truth for branching strategy resolution.

        Priority (highest to lowest):
            1. values["branching_strategy"]  — runtime override
            2. branching_strategy constructor arg
            3. values["use_ryan_foster_branching"] legacy bool
            4. use_ryan_foster constructor bool (deprecated)
        """
        if "branching_strategy" in values:
            return str(values["branching_strategy"])
        if branching_strategy != "edge":  # non-default means explicitly set
            return branching_strategy
        if values.get("use_ryan_foster_branching", False):
            return "ryan_foster"
        if use_ryan_foster:
            return "ryan_foster"
        return "edge"

    def _get_statistics(self) -> Dict[str, Any]:
        """Collect and return solver statistics."""
        lp_b = self.lp_bound if self.lp_bound is not None else 0.0
        ip_s = self.ip_solution if self.ip_solution is not None else 0.0
        gap = abs(lp_b - ip_s) / max(abs(lp_b), 1e-10)

        stats: Dict[str, Any] = {
            "num_iterations": self.num_iterations,
            "num_columns_generated": self.num_columns_generated,
            "lp_bound": self.lp_bound,
            "ip_solution": self.ip_solution,
            "gap": gap,
            "branching_strategy": self.params.branching_strategy,
            "use_ng_routes": self.params.use_ng_routes,
            "ng_neighborhood_size": self.params.ng_neighborhood_size,
            "proven_optimal": self.proven_optimal,
        }
        if self.tree is not None:
            stats.update(self.tree.get_statistics())
        return stats
