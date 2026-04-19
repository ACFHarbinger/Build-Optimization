"""
ILS-RVND-SP solver.

Combines the Iterated Local Search (ILS) metaheuristic with Randomized
Variable Neighborhood Descent (RVND) and an exact Set Partitioning (SP) formulation.

Reference:
    Subramanian et al. "A hybrid algorithm for a class of vehicle routing problems",
    Computers & Operations Research, 2013.
"""

import copy
import logging
import time
from random import Random
from typing import Callable, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.policies.helpers.local_search.local_search_aco import (
    ACOLocalSearch,
)
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
)
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params import (
    ILSRVNDSPParams,
)
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd import (
    RVND,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

logger = logging.getLogger(__name__)


class ILSRVNDSPSolver:
    """
    ILS-RVND-SP solver executing ILS with RVND and resolving via Set Partitioning.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ILSRVNDSPParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the ILS-RVND-SP solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes (demands/profits).
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Parameters for the algorithm.
            mandatory_nodes: List of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()
        self.random = Random(params.seed) if params.seed is not None else Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Keep track of unique valid routes explored
        # A route is mapped to a tuple for correct hashing
        self.route_pool: Set[Tuple[int, ...]] = set()

        self.ls_manager = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=KSACOParams(
                local_search_iterations=self.params.local_search_iterations,
                vrpp=self.params.vrpp,
                profit_aware_operators=self.params.profit_aware_operators,
                seed=self.params.seed,
            ),
        )

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def calculate_profit(self, routes: List[List[int]]) -> float:
        """Calculate network profit (revenue - cost)."""
        cost = self.calculate_cost(routes)
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - cost

    def _add_to_pool(self, routes: List[List[int]], pool: Optional[Set[Tuple[int, ...]]] = None):
        """
        Standardize and hash routes to add to the unique pool with canonical representation.

        Routes [1, 2, 3] and [3, 2, 1] are geographically identical (same tour, opposite direction).
        To prevent symmetric duplicates from bloating the MIP pool, we enforce a canonical
        orientation: the route endpoint with the smaller node ID is always placed first.

        This reduces the MIP pool size by ~50% and significantly speeds up Gurobi solving.
        """
        target_pool = pool if pool is not None else self.route_pool
        for route in routes:
            if not route:
                continue

            # Enforce canonical representation based on endpoint node IDs
            # Orient route so the smaller endpoint ID is always first (or keep single-node route as-is)
            canonical_route = tuple(route if route[0] < route[-1] else route[::-1]) if len(route) > 1 else tuple(route)

            target_pool.add(canonical_route)

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Random perturbation for ILS as specified in Subramanian et al. (2013).

        Performs 2-5 random moves. Each move is either:
        - Swap(1,1): Swap one random node from r1 with one random node from r2.
        - Shift(1,1): Relocate one random node from r1 to a random position in r2.
        """
        new_routes = [r[:] for r in routes]
        n_moves = self.random.randint(2, 5)

        for _ in range(n_moves):
            # Only perturb if we have at least 2 non-empty routes for inter-route moves
            active_routes = [i for i, r in enumerate(new_routes) if len(r) > 0]
            if len(active_routes) < 2:
                # Fallback: if only one route, just do a random intra-swap if possible
                if len(active_routes) == 1:
                    ridx = active_routes[0]
                    if len(new_routes[ridx]) >= 2:
                        i, j = self.random.sample(range(len(new_routes[ridx])), 2)
                        new_routes[ridx][i], new_routes[ridx][j] = new_routes[ridx][j], new_routes[ridx][i]
                continue

            if self.random.random() < 0.5:
                # Swap(1,1)
                r1_idx, r2_idx = self.random.sample(active_routes, 2)
                p1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                p2 = self.random.randint(0, len(new_routes[r2_idx]) - 1)
                new_routes[r1_idx][p1], new_routes[r2_idx][p2] = new_routes[r2_idx][p2], new_routes[r1_idx][p1]
            else:
                # Shift(1,1) (Relocate)
                # Note: Shift(1,1) in paper terminology usually means a relocate move
                r1_idx = self.random.choice(active_routes)
                r2_idx = self.random.choice(range(len(new_routes)))  # Can shift to any route
                p1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                node = new_routes[r1_idx].pop(p1)
                p2 = self.random.randint(0, len(new_routes[r2_idx])) if new_routes[r2_idx] else 0
                new_routes[r2_idx].insert(p2, node)

        return new_routes

    def build_initial_solution(self) -> List[List[int]]:
        """Construct initial solution using standard greedy heuristic."""
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def solve_set_partitioning(
        self, pool: Optional[Set[Tuple[int, ...]]] = None
    ) -> Tuple[List[List[int]], float, float]:
        """
        Solve the Set Packing Problem exactly using Gurobi over the route pool.

        For VRPP (Vehicle Routing Problem with Profits):
        - Mandatory nodes must be visited exactly once (equality constraint)
        - Optional nodes can be visited at most once (inequality constraint <= 1)
        - Unvisited nodes contribute 0 to profit (no penalty)
        - Objective: maximize total profit (revenue - routing cost)

        This is a Set Packing formulation adapted for VRPP, not strict Set Partitioning.
        """
        target_pool = list(pool) if pool is not None else list(self.route_pool)
        n_routes = len(target_pool)

        if n_routes == 0:
            return [], 0.0, 0.0

        logger.info(f"ILS-RVND-SP solving Set Partitioning over {n_routes} unique routes.")

        # Precompute costs and node incidence
        route_costs = np.zeros(n_routes)
        route_profits = np.zeros(n_routes)

        # Map node to the indices of routes that contain it
        node_to_routes: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for i, route in enumerate(target_pool):
            cost = self.dist_matrix[0][route[0]]
            for j in range(len(route) - 1):
                cost += self.dist_matrix[route[j]][route[j + 1]]
            cost += self.dist_matrix[route[-1]][0]
            route_costs[i] = cost * self.C

            revenue = sum(self.wastes.get(n, 0.0) * self.R for n in route)
            route_profits[i] = revenue - route_costs[i]

            for node in route:
                node_to_routes[node].append(i)

        # Build Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("ILSRVNDSP_SetPartitioning", env=env)

        model.setParam("TimeLimit", getattr(self.params, "mip_time_limit", 60.0))
        model.setParam("MIPGap", getattr(self.params, "sp_mip_gap", 0.01))

        # Variables: x[i] = 1 if route i is selected
        x = model.addVars(n_routes, vtype=GRB.BINARY, name="x")

        # Maximize Profit (Revenue - Cost)
        model.setObjective(gp.quicksum(route_profits[i] * x[i] for i in range(n_routes)), GRB.MAXIMIZE)

        # Constraints: Mandatory nodes visited exactly once
        mandatory_set = set(self.mandatory_nodes)
        for node in mandatory_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) == 1, name=f"Mandatory_{node}")

        # Constraints: Optional nodes visited at most once
        optional_set = set(self.nodes) - mandatory_set
        for node in optional_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) <= 1, name=f"Optional_{node}")

        # Constraints: Vehicle fleet limit (Subramanian 2013)
        max_v = getattr(self.params, "max_vehicles", 0)
        if max_v > 0:
            model.addConstr(gp.quicksum(x[i] for i in range(n_routes)) <= max_v, name="FleetLimit")

        model.optimize()

        if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} and model.SolCount > 0:
            best_routes = []
            for i in range(n_routes):
                if x[i].X > 0.5:
                    best_routes.append(list(target_pool[i]))

            best_cost = self.calculate_cost(best_routes)
            best_profit = self.calculate_profit(best_routes)
            return best_routes, best_profit, best_cost

        logger.warning("SP failed to find a valid combination. Falling back to best ILS solution.")
        return [], -float("inf"), float("inf")

    def run_ils_rvnd(
        self,
        initial_routes: List[List[int]],
        max_iterations: int,
        max_ils_iterations: int,
        target_pool: Set[Tuple[int, ...]],
        tolerance: float,
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute ILS-RVND metaheuristic iterations."""
        best_routes = copy.deepcopy(initial_routes)
        best_profit = self.calculate_profit(best_routes)
        current_profit = best_profit

        # Setup modular acceptance criterion
        self.params.acceptance_criterion.setup(best_profit)

        self._add_to_pool(best_routes, target_pool)

        for iteration in range(max_iterations):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break

            iter_routes = copy.deepcopy(best_routes)

            for ils_idx in range(max_ils_iterations):
                if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                    break

                # Perturb current solution and apply RVND intensification
                perturbed = self._perturb(iter_routes)
                ls_routes = rvnd.apply(perturbed)
                ls_profit = self.calculate_profit(ls_routes)

                # Dynamic tolerance pool addition
                if len(ls_routes) > 0 and (best_profit - ls_profit) / abs(best_profit + 1e-9) <= tolerance:
                    self._add_to_pool(ls_routes, target_pool)

                # ILS Acceptance Criterion:
                # Delegate decision (improve only, SA, GD, etc. based on injection)
                is_accepted, _ = self.params.acceptance_criterion.accept(
                    current_obj=current_profit,
                    candidate_obj=ls_profit,
                    iteration=iteration * max_ils_iterations + ils_idx,
                    max_iterations=max_iterations * max_ils_iterations,
                )

                if is_accepted:
                    iter_routes = copy.deepcopy(ls_routes)
                    current_profit = ls_profit

                # Global best update: strict improvement only (always)
                if ls_profit > best_profit + 1e-6:
                    best_routes = copy.deepcopy(ls_routes)
                    best_profit = ls_profit

                # Step criterion
                self.params.acceptance_criterion.step(
                    current_obj=current_profit,
                    candidate_obj=ls_profit,
                    accepted=is_accepted,
                    iteration=iteration * max_ils_iterations + ils_idx,
                )

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                score=3 if best_profit > current_profit else 0,
            )

        return best_routes, best_profit

    def _run_strategy_a(
        self,
        initial_solution: Optional[List[List[int]]],
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute Strategy A for smaller instances (n <= N)."""
        global_best_routes = []
        global_best_profit = -float("inf")

        max_restarts = getattr(self.params, "max_restarts", 10)
        max_iter_a = getattr(self.params, "MaxIter_a", 50)
        iter_count = max_iter_a if max_iter_a > 0 else max_restarts

        # MaxIterILS-a = n + 0.5 * v (Subramanian 2013)
        init_routes_for_v = initial_solution if initial_solution else self.build_initial_solution()
        v_approx = max(1, len(init_routes_for_v))
        ils_count = int(self.n_nodes + 0.5 * v_approx)

        tolerance = getattr(self.params, "TDev_a", 0.05)

        for _ in range(iter_count):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break
            init_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
            iter_best_routes, iter_best_profit = self.run_ils_rvnd(
                init_routes, 1, ils_count, self.route_pool, tolerance, start_time, rvnd
            )
            if iter_best_profit > global_best_profit:
                global_best_profit = iter_best_profit
                global_best_routes = iter_best_routes

        if getattr(self.params, "use_set_partitioning", True) and self.route_pool:
            sp_routes, sp_profit, _ = self.solve_set_partitioning(self.route_pool)
            if sp_routes and sp_profit > global_best_profit:
                global_best_routes = sp_routes
                global_best_profit = sp_profit

        return global_best_routes, global_best_profit

    def _run_strategy_b(
        self,
        initial_solution: Optional[List[List[int]]],
        start_time: float,
        rvnd: RVND,
    ) -> Tuple[List[List[int]], float]:
        """Execute Strategy B for larger instances (n > N)."""
        global_best_routes = []
        global_best_profit = -float("inf")

        max_restarts = getattr(self.params, "max_restarts", 10)
        max_iter_b = getattr(self.params, "MaxIter_b", 100)
        max_iter_ils = getattr(self.params, "max_iter_ils", 50)
        max_ils_b = getattr(self.params, "MaxIterILS_b", 2000)
        a_ratio = getattr(self.params, "A", 11.0)
        tdev_b = getattr(self.params, "TDev_b", 0.005)
        mip_time_limit = getattr(self.params, "mip_time_limit", 60.0)

        iter_count = max_iter_b if max_iter_b > 0 else max_restarts
        ils_count = max_ils_b if max_ils_b > 0 else max_iter_ils

        for _ in range(iter_count):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break

            init_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
            num_vehicles_approx = len(init_routes) if len(init_routes) > 0 else 1

            tolerance = tdev_b if (self.n_nodes / num_vehicles_approx) < a_ratio else 1.0

            temp_pool: Set[Tuple[int, ...]] = set()
            iter_best_routes, iter_best_profit = self.run_ils_rvnd(
                init_routes, 1, ils_count, temp_pool, tolerance, start_time, rvnd
            )

            combined_pool = self.route_pool.union(temp_pool)

            if getattr(self.params, "use_set_partitioning", True) and combined_pool:
                sp_time = time.process_time()
                sp_routes, sp_profit, _ = self.solve_set_partitioning(combined_pool)
                sp_duration = time.process_time() - sp_time

                if sp_routes and sp_profit > iter_best_profit:
                    iter_best_routes = sp_routes
                    iter_best_profit = sp_profit

                if not (sp_routes and sp_profit > iter_best_profit):
                    temp_pool.clear()

                if (self.n_nodes / num_vehicles_approx) < a_ratio:
                    if sp_duration > mip_time_limit:
                        tolerance -= 0.1 * tdev_b
                    elif sp_duration < 1.0:
                        tolerance += 0.1 * tdev_b

            if iter_best_profit > global_best_profit:
                global_best_profit = iter_best_profit
                global_best_routes = iter_best_routes
                self.route_pool.update(temp_pool)

        return global_best_routes, global_best_profit

    def _create_rvnd_operators(self) -> List[Callable]:
        """
        Create atomic RVND operator wrappers for distinct local search neighborhoods.

        This is critical for proper RVND functionality. Each operator must target
        a specific, distinct neighborhood type. RVND's randomized neighborhood list
        mechanics only work when operators are truly independent.

        Returns:
            List of callable operators, each targeting a specific neighborhood.
        """
        operators = []

        # Define the atomic neighborhoods to explore
        # Each operator will run local search restricted to ONE specific move type
        # Exact 10 neighborhoods specified in Subramanian et al. (2013)
        neighborhoods = [
            "intra_swap",  # Exchange (intra)
            "intra_2opt",  # 2-opt (intra)
            "intra_or_opt",  # Or-opt2 + Or-opt3 (intra)
            "inter_relocate",  # Shift(1,0) (inter)
            "shift_2_0",  # Shift(2,0) (inter) [NEW]
            "inter_swap",  # Swap(1,1) (inter)
            "swap_2_1",  # Swap(2,1) (inter) [NEW]
            "swap_2_2",  # Swap(2,2) (inter) [NEW]
            "cross",  # Cross / Suffix exchange (inter) [NEW]
            "unrouted_insert",  # VRPP specific insertion (extension)
        ]

        # Create a wrapper for each neighborhood
        for neighborhood_type in neighborhoods:

            def make_wrapper(nh_type=neighborhood_type):
                """Factory to capture neighborhood_type in closure."""

                def operator_wrapper(routes: List[List[int]]) -> Tuple[List[List[int]], bool]:
                    """
                    RVND operator that applies ONLY the specified neighborhood.

                    Args:
                        routes: Current solution routes.

                    Returns:
                        (optimized_routes, improvement_found)
                    """
                    if not routes:
                        return routes, False

                    # Calculate initial profit
                    initial_profit = self.calculate_profit(routes)

                    # Run local search with ONLY the specific neighborhood
                    optimized_routes = self.ls_manager.optimize(routes, target_neighborhood=nh_type)

                    # Calculate final profit
                    final_profit = self.calculate_profit(optimized_routes)

                    # Check if this neighborhood found an improvement
                    improved = final_profit > initial_profit + 1e-6

                    return optimized_routes, improved

                return operator_wrapper

            operators.append(make_wrapper())

        return operators

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Main ILS-RVND-SP execution loop with dual strategy handling."""
        start_time = time.process_time()

        # Create RVND with local search operators
        operators = self._create_rvnd_operators()
        rvnd = RVND(operators=operators, rng=self.random)

        n_limit = getattr(self.params, "N", 150)

        if self.n_nodes <= n_limit:
            global_best_routes, global_best_profit = self._run_strategy_a(initial_solution, start_time, rvnd)
        else:
            global_best_routes, global_best_profit = self._run_strategy_b(initial_solution, start_time, rvnd)

        best_cost = self.calculate_cost(global_best_routes)
        return global_best_routes, global_best_profit, best_cost
