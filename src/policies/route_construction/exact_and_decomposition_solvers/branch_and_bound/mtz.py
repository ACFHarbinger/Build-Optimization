"""
Branch-and-Bound (BB) solver core engine.

Standard Best-Bound-First Branch-and-Bound algorithm based on LP relaxations.
While Land & Doig (1960) established the foundational concept of Branch-and-Bound
and spatial branching, this modern implementation utilizes cold/warm-start LP
resolves at each node (via Gurobi) rather than the original parametric systematic
hyperplane shifts.

The solver manages a search tree to ensure global optimality for VRPP/CWC VRP problems.

**REFACTORED**: Uses the Miller-Tucker-Zemlin (MTZ) compact formulation to ensure
subtour elimination strictly within the LP relaxations at each node.

**PERFORMANCE OPTIMIZATIONS**:
1. Persistent Gurobi model (no rebuilding at each node)
2. Maximization objective (direct profit maximization)
3. True Strong Branching (LP-based variable selection)

Reference:
    Land, A. H., & Doig, A. (1960). "An automatic method for solving discrete
    programming problems". Econometrica, 28(3), 497-520.
"""

import heapq
import itertools
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.helpers.solvers_and_matheuristics.common.node import Node
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .params import BBParams


class BBSolver:
    """
    Exact Branch-and-Bound engine for Vehicle Routing Problems.

    This engine is designed to solve the Profit-Maximizing or Waste-Collecting
    Vehicle Routing Problem with high precision, utilizing automated
    variable selection and node pruning based on deterministic bounds.

    The solver maintains an 'incumbent' solution (the best integer solution
    found so far) and stops when the search tree is exhausted or the
    specified MIP gap is reached.

    **ARCHITECTURAL JUSTIFICATION**:
    This custom solver is engineered specifically for **full observability of the
    search state**. Exposing the B&B tree entirely in Python is a necessary
    architectural choice for integrating machine learning models (e.g., injecting
    Graph Neural Networks for state evaluation or imitation learning for variable
    selection heuristics), which is heavily restricted by standard commercial
    solver callback systems.

    **REFACTORED DESIGN**:
    - Persistent Gurobi model stored in self.model (initialized once)
    - Fast node evaluation via variable bound updates (no model rebuilding)
    - Maximization objective (profit maximization, not cost minimization)
    - True Strong Branching using LP solves
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        time_limit: float = 60.0,
        mip_gap: float = 0.01,
        branching_strategy: str = "strong",
        strong_branching_limit: int = 5,
        mandatory_indices: Optional[Set[int]] = None,
        env: Optional[gp.Env] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the Branch-and-Bound solver.

        Args:
            dist_matrix: Symmetric matrix representing edge costs.
            wastes: Mapping customer IDs to their respective fill levels.
            capacity: Physical payload limit for the routing model.
            R: Revenue coefficient per unit of collected waste.
            C: Cost coefficient per unit of distance traveled.
            values: Merged configuration dictionary from hydra and adapter.
            mandatory_indices: Set of customer nodes that MUST be included in routes.
            env: Optional Gurobi environment for resource management.
            seed: Optional random seed.
            recorder: Optional telemetry recorder for state tracking.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.branching_strategy = branching_strategy
        self.strong_branching_limit = strong_branching_limit
        self.mandatory_indices = mandatory_indices or set()
        self.env = env
        self.recorder = recorder

        self.num_nodes = len(dist_matrix)
        self.nodes_range = range(self.num_nodes)
        self.customers = list(range(1, self.num_nodes))

        # CHANGE: Maximization problem, so incumbent starts at -inf
        self.incumbent_obj = -float("inf")
        self.incumbent_routes: List[List[int]] = []
        self.start_time = 0.0

        # NEW: Persistent model and variable dictionaries
        self.model: Optional[gp.Model] = None
        self.x: Dict[Tuple[int, int], gp.Var] = {}
        self.y: Dict[int, gp.Var] = {}
        self.u: Dict[int, gp.Var] = {}

        # Initialize the base model once
        self._initialize_base_model()

    def _initialize_base_model(self) -> None:
        """
        Construct the persistent Linear Programming model with MTZ formulation.

        This method is called once during __init__ to create the base Gurobi model.
        All variables are defined as CONTINUOUS with bounds [0.0, 1.0].
        The model is then reused throughout the search by updating variable bounds.

        Uses Miller-Tucker-Zemlin (MTZ) load-based formulation for subtour elimination.
        MTZ provides a compact formulation suitable for LP relaxations without callbacks.
        The load variables (u[i]) track accumulated waste, naturally preventing subtours.

        **OBJECTIVE**: Maximize (Revenue - Travel Cost)
        """
        self.model = gp.Model("BB_Persistent", env=self.env) if self.env else gp.Model("BB_Persistent")
        self.model.setParam("OutputFlag", 0)  # Silence Gurobi output for sub-solves
        self.model.setParam("Threads", 1)  # Single-threaded sub-solves for better control

        # --- Decision Variables (CONTINUOUS for LP relaxation) ---

        # x[i,j]: Edge usage (CONTINUOUS [0,1] for LP relaxation)
        for i in self.nodes_range:
            for j in self.nodes_range:
                if i != j:
                    self.x[i, j] = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

        # y[i]: Node visit (CONTINUOUS [0,1] for LP relaxation)
        for i in self.customers:
            lb = 1.0 if i in self.mandatory_indices else 0.0
            self.y[i] = self.model.addVar(lb=lb, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{i}")

        # u[i]: MTZ load variables for subtour elimination
        for i in self.customers:
            self.u[i] = self.model.addVar(
                lb=self.wastes.get(i, 0.0), ub=self.capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}"
            )

        # --- Objective: Maximize (Revenue - Travel Cost) ---
        revenue = quicksum(self.y[i] * self.wastes.get(i, 0) * self.R for i in self.customers)
        travel_cost = quicksum(self.dist_matrix[i][j] * self.C * self.x[i, j] for (i, j) in self.x)
        self.model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

        # --- Base Constraints (Flow and Integrity) ---
        for i in self.customers:
            self.model.addConstr(
                quicksum(self.x[i, j] for j in self.nodes_range if (i, j) in self.x) == self.y[i],
                name=f"out_{i}",
            )
            self.model.addConstr(
                quicksum(self.x[j, i] for j in self.nodes_range if (j, i) in self.x) == self.y[i],
                name=f"in_{i}",
            )

        # Depot flow constraint
        self.model.addConstr(quicksum(self.x[0, j] for j in self.customers if (0, j) in self.x) <= 1, name="depot_out")
        self.model.addConstr(
            quicksum(self.x[j, 0] for j in self.customers if (j, 0) in self.x)
            == quicksum(self.x[0, j] for j in self.customers if (0, j) in self.x),
            name="depot_balance",
        )

        # Capacity constraint
        self.model.addConstr(
            quicksum(self.wastes.get(i, 0) * self.y[i] for i in self.customers) <= self.capacity,
            name="capacity",
        )

        # --- MTZ Subtour Elimination Constraints (Load-Based) ---
        # u[i] + waste[j] <= u[j] + Capacity * (1 - x[i,j])
        # This prevents any closed loops that do not originate from the depot
        for i in self.customers:
            for j in self.customers:
                if i != j and (i, j) in self.x:
                    self.model.addConstr(
                        self.u[i] + self.wastes.get(j, 0.0) <= self.u[j] + self.capacity * (1 - self.x[i, j]),
                        name=f"mtz_{i}_{j}",
                    )

    def _evaluate_node(self, node: Node) -> float:
        """
        Evaluate a search node by updating variable bounds and solving the LP relaxation.

        This method reuses the persistent self.model by:
        1. Resetting all variable bounds to [0.0, 1.0]
        2. Applying the fixed variable values from the node
        3. Solving the LP relaxation
        4. Returning the objective value

        Args:
            node: The current search node containing fixed variable states.

        Returns:
            The LP relaxation objective value, or -inf if infeasible.
        """
        # Step 1: Reset all variable bounds to [0.0, 1.0]
        for (_i, _j), var in self.x.items():
            var.LB = 0.0
            var.UB = 1.0

        for i, var in self.y.items():
            lb = 1.0 if i in self.mandatory_indices else 0.0
            var.LB = lb
            var.UB = 1.0

        # Reset u (MTZ load) variable bounds to initialization values
        for i, var in self.u.items():
            var.LB = self.wastes.get(i, 0.0)
            var.UB = self.capacity

        # Step 2: Apply fixed variable values from the node
        for (i, j), val in node.fixed_x.items():
            self.x[i, j].LB = float(val)
            self.x[i, j].UB = float(val)

        for i, val in node.fixed_y.items():
            self.y[i].LB = float(val)
            self.y[i].UB = float(val)

        # Step 3: Solve the LP relaxation
        self.model.optimize()  # type: ignore[union-attr]

        # Step 4: Return the objective value
        if self.model.Status == GRB.OPTIMAL:  # type: ignore[union-attr]
            return self.model.ObjVal  # type: ignore[union-attr]
        else:
            return -float("inf")

    def _is_integer(self, val: float, tol: float = 1e-6) -> bool:
        """Heuristic check for integer feasibility of a continuous variable."""
        return abs(val - round(val)) < tol

    def _get_branching_variable(self) -> Optional[Tuple[str, Any]]:
        """
        Identify the most suitable fractional variable to branch upon.

        Supported strategies:
        - 'strong': True Strong Branching using LP solves.
        - 'most_fractional': Maximize the uncertainty (closest to 0.5).
        - 'least_fractional': Minimize the uncertainty (closest to integer bounds).

        Returns:
            A tuple identifying the variable ('x'|'y', Index) or None if feasible.
        """
        fractional_vars = []

        for i_j, var in self.x.items():
            if not self._is_integer(var.X):
                fractional_vars.append(("x", i_j, var.X))

        # Note: We only collect fractional x variables. Branching on y is redundant
        # because the flow constraints sum_j x[i,j] = y[i] and sum_j x[j,i] = y[i]
        # ensure that if y[i] is fractional, at least one incident x variable
        # must also be fractional. Branching on x directly is more efficient.

        if not fractional_vars:
            return None

        if self.branching_strategy == "strong":
            return self._strong_branching(fractional_vars)

        if self.branching_strategy == "most_fractional":
            return sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[0][:2]

        return sorted(fractional_vars, key=lambda v: min(v[2], 1 - v[2]))[0][:2]

    def _strong_branching(self, fractional_vars: List[Tuple]) -> Tuple[str, Any]:
        """
        True Strong Branching: choose variable that causes maximum bound degradation.

        For each candidate variable:
        1. Fix the variable to 0.0, solve LP, record objective z_down
        2. Fix the variable to 1.0, solve LP, record objective z_up
        3. Restore the variable bounds to [0.0, 1.0]

        The score is computed as: min(root_obj - z_down, root_obj - z_up)
        This measures the worst-case degradation in the objective bound.

        Args:
            fractional_vars: Candidate variables (type, index, value).

        Returns:
            The variable (type, index) that yielded the highest score.
        """
        # Limit the number of candidates for performance
        candidates = sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[: self.strong_branching_limit]

        # Store current objective as the baseline
        root_obj = self.model.ObjVal  # type: ignore[union-attr]

        best_var = candidates[0][:2]
        best_score = -float("inf")

        for var_type, var_idx, _val in candidates:
            # Get the variable reference
            var = self.x[var_idx] if var_type == "x" else self.y[var_idx]

            # Save original bounds
            orig_lb = var.LB
            orig_ub = var.UB

            # Test 1: Fix variable to 0
            var.LB = 0.0
            var.UB = 0.0
            self.model.optimize()  # type: ignore[union-attr]
            z_down = self.model.ObjVal if self.model.Status == GRB.OPTIMAL else -float("inf")  # type: ignore[union-attr]

            # Test 2: Fix variable to 1
            var.LB = 1.0
            var.UB = 1.0
            self.model.optimize()  # type: ignore[union-attr]
            z_up = self.model.ObjVal if self.model.Status == GRB.OPTIMAL else -float("inf")  # type: ignore[union-attr]

            # Restore original bounds
            var.LB = orig_lb
            var.UB = orig_ub

            # Score: degradation from root_obj (for maximization, degradation = decrease)
            # We want the variable that causes the largest degradation in the worst direction
            degradation_down = root_obj - z_down if z_down > -float("inf") else float("inf")
            degradation_up = root_obj - z_up if z_up > -float("inf") else float("inf")
            score = min(degradation_down, degradation_up)

            if score > best_score:
                best_score = score
                best_var = (var_type, var_idx)

        # Restore the node's LP state by re-solving after final bound restoration.
        # This re-solve is correct because all candidates had their bounds
        # restored to orig_lb/orig_ub (the node's bounds) within the loop or
        # immediately after their probes, so the model state now correctly
        # reflects the original node subproblem.
        self.model.optimize()  # type: ignore[union-attr]

        return best_var

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Perform the Branch-and-Bound search to find the optimal integer routes.

        Implements the core manual B&B loop using LP-based relaxations (Best-Bound-First):
        1. Initialize priority queue with root node
        2. Loop while queue is not empty:
           - Pop node with best bound
           - Prune by bound if node cannot improve incumbent
           - Evaluate LP relaxation with fixed variables
           - Prune by infeasibility or bound
           - Check integrality:
             * If integer and valid → update incumbent
             * If fractional → branch

        **MAXIMIZATION LOGIC**:
        - Priority queue uses negative bounds for max-heap behavior
        - Prune if node.bound <= incumbent_obj * (1 + mip_gap)

        Returns:
            A tuple of (Optimal Integer Routes, Total Objective Value).
        """
        self.start_time = time.perf_counter()

        # Initialize tie-breaker counter to ensure deterministic heap ordering
        counter = itertools.count()

        # Step 1: Initialize priority queue with root node
        # Root node has no fixed variables
        root_node = Node(bound=float("inf"))  # Upper bound unknown until LP is solved
        queue = [(-float("inf"), next(counter), root_node)]  # Priority uses -inf until first LP

        nodes_explored = 0

        # Step 2: Main B&B loop
        while queue:
            # Check time limit
            if time.perf_counter() - self.start_time >= self.time_limit:
                break

            # Pop node with best bound (for maximization, we negate the priority)
            _, _, current_node = heapq.heappop(queue)
            nodes_explored += 1

            # Pruning by Bound (before LP solve)
            # For maximization: if node.bound <= incumbent_obj * (1 + mip_gap), prune
            gap_threshold = self.incumbent_obj + abs(self.incumbent_obj) * self.mip_gap
            if self.incumbent_obj > -float("inf") and current_node.bound <= gap_threshold:
                continue

            # Evaluate LP relaxation
            current_obj = self._evaluate_node(current_node)

            # Pruning by Infeasibility
            if current_obj == -float("inf"):
                continue

            # Pruning by Bound (after LP solve)
            # For maximization: if current_obj <= incumbent_obj * (1 + mip_gap), prune
            gap_threshold = self.incumbent_obj + abs(self.incumbent_obj) * self.mip_gap
            if self.incumbent_obj > -float("inf") and current_obj <= gap_threshold:
                continue

            # Integrality Check
            branch_var = self._get_branching_variable()

            if branch_var is None:
                # Solution is integer-feasible
                # Verify no subtours exist (should be prevented by MTZ)
                routes = self._extract_routes()

                # Update incumbent (for maximization, keep the maximum)
                if routes and self._is_valid_solution(routes) and current_obj > self.incumbent_obj:
                    self.incumbent_obj = current_obj
                    self.incumbent_routes = routes
            else:
                # Branching - Variable is fractional
                var_type, var_idx = branch_var

                # Create two child nodes

                # Left Child (Exclude: variable = 0)
                left_child = Node(
                    bound=current_obj,
                    fixed_x=current_node.fixed_x.copy(),
                    fixed_y=current_node.fixed_y.copy(),
                    depth=current_node.depth + 1,
                )
                if var_type == "x":
                    left_child.fixed_x[var_idx] = 0
                else:
                    left_child.fixed_y[var_idx] = 0

                # For max-heap, push with negative priority and tie-breaker
                heapq.heappush(queue, (-current_obj, next(counter), left_child))

                # Right Child (Enforce: variable = 1)
                right_child = Node(
                    bound=current_obj,
                    fixed_x=current_node.fixed_x.copy(),
                    fixed_y=current_node.fixed_y.copy(),
                    depth=current_node.depth + 1,
                )
                if var_type == "x":
                    right_child.fixed_x[var_idx] = 1
                else:
                    right_child.fixed_y[var_idx] = 1

                # For max-heap, push with negative priority and tie-breaker
                heapq.heappush(queue, (-current_obj, next(counter), right_child))

        # Telemetry updates
        if self.recorder:
            self.recorder.record(
                engine="bb_lp_relaxation",
                nodes_explored=nodes_explored,
                obj_val=self.incumbent_obj,
                time=time.perf_counter() - self.start_time,
            )

        return self.incumbent_routes, self.incumbent_obj

    def _is_valid_solution(self, routes: List[List[int]]) -> bool:
        """
        Validate that a solution is feasible (no subtours, respects capacity).

        Args:
            routes: List of routes to validate.

        Returns:
            True if solution is valid, False otherwise.
        """
        # Check capacity constraint
        for route in routes:
            load = sum(self.wastes.get(i, 0) for i in route)
            if load > self.capacity:
                return False

        # Check that all nodes are visited at most once
        visited = set()
        for route in routes:
            for node in route:
                if node in visited:
                    return False  # Node visited multiple times
                visited.add(node)

        # Check mandatory nodes
        return all(node in visited for node in self.mandatory_indices)

    def _extract_routes(self) -> List[List[int]]:
        """
        Convert sparse binary edge variables into ordered route sequences.

        Traverses the solution adjacency graph starting from the depot (0).

        Returns:
            List of routes (each route is a list of customer node IDs).
        """
        edges = [e for e, v in self.x.items() if v.X > 0.5]
        adj: Dict[int, List[int]] = {i: [] for i in self.nodes_range}
        for i, j in edges:
            adj[i].append(j)

        routes = []
        for start_node in adj[0]:
            route = [start_node]
            curr = start_node
            visited = {0, start_node}
            while curr != 0 and adj[curr]:
                next_node = adj[curr][0]
                if next_node in visited and next_node != 0:
                    break  # Cycle detected, stop
                if next_node != 0:
                    route.append(next_node)
                    visited.add(next_node)
                curr = next_node
                if curr == 0:
                    break
            if route:
                routes.append(route)
        return routes


def run_bb_mtz(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: Optional[BBParams] = None,
    mandatory_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
    **kwargs: Any,
) -> Tuple[List[List[int]], float]:
    """
    Dispatcher entry point for the MTZ-formulation Branch-and-Bound solver.

    Uses Miller-Tucker-Zemlin (MTZ) compact formulation with load variables
    for subtour elimination in LP relaxations.

    Args:
        dist_matrix: Symmetric distance matrix (n x n).
        wastes: Mapping of customer IDs to fill levels/profits.
        capacity: Vehicle payload capacity.
        R: Revenue coefficient per unit collected.
        C: Cost coefficient per unit distance.
        params: Standardized BB parameters.
        mandatory_indices: Set of mandatory customer nodes.
        env: Optional Gurobi environment for resource management.
        recorder: Optional telemetry recorder.

    Returns:
        Tuple of (routes, objective_value).
    """
    if params is None:
        params = BBParams()

    solver = BBSolver(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        time_limit=params.time_limit,
        mip_gap=params.mip_gap,
        branching_strategy=params.branching_strategy,
        strong_branching_limit=params.strong_branching_limit,
        mandatory_indices=mandatory_indices,
        env=env,
        recorder=recorder,
    )
    return solver.solve()
