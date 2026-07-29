"""
Kernel Search matheuristic solver for VRPP.

This module implements the core Gurobi-based Kernel Search algorithm.
The framework generalizes decomposition techniques by restricting binary variables
to a promising subset (the Kernel) and iteratively expanding it with additional sets (Buckets).

Reference:
    Angelelli, R., Dell'Amico, M., & Martello, S. (2010). "A Kernel Search Algorithm for the
    Vehicle Routing Problem with Pickups and Deliveries". Networks, 56(4), 297–307.
    Angelelli, R., & Mansini, R. (2010). "Kernel Search: a new heuristic framework for
    portfolio selection".
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _check_route_capacity(model, G, x_vars, component):
    """
    Check if any routes from depot violate capacity constraints and add lazy cuts if needed.

    Args:
        model: Gurobi model instance
        G: NetworkX graph of active edges
        x_vars: Dictionary of edge variables
        component: Set of nodes in the current component (must contain depot 0)
    """
    if 0 not in component:
        return

    # Find the specific paths originating from the depot within this component
    depot_edges = [j for j in G.neighbors(0)]
    for start_node in depot_edges:
        # Trace the route
        route_nodes = set()
        route_edges = []
        curr = start_node
        prev = 0
        route_waste = 0.0

        while curr != 0:
            route_nodes.add(curr)
            route_edges.append((prev, curr) if (prev, curr) in x_vars else (curr, prev))
            route_waste += model._wastes.get(curr, 0.0)

            # Move to next node
            neighbors = [n for n in G.neighbors(curr) if n != prev]
            if not neighbors:
                break  # Dead end
            prev = curr
            curr = neighbors[0]

        route_edges.append((prev, 0) if (prev, 0) in x_vars else (0, prev))

        # Add a capacity cut if the route violates the limit
        if route_waste > model._capacity:
            route_vars = [x_vars[e] for e in route_edges if e in x_vars]
            if route_vars:
                model.cbLazy(quicksum(route_vars) <= len(route_vars) - 1)


def _dfj_subtour_elimination_callback(model, where):
    """
    Gurobi callback to dynamically add Dantzig-Fulkerson-Johnson (DFJ) subtour elimination cuts.

    This callback is triggered at integer solutions (MIPSOL) and uses NetworkX to detect
    connected components. If a component S does not contain the depot (node 0), it injects
    a lazy constraint enforcing that at least one edge must leave the subtour:

        sum_{i in S, j in S, i != j} x[i,j] <= |S| - 1

    Mathematical Foundation:
        The DFJ formulation provides an exponentially-sized family of constraints that
        completely eliminate all subtours. Unlike MTZ, which uses auxiliary continuous
        variables with weak LP bounds, DFJ cuts directly target the combinatorial structure
        of the tour polytope, yielding significantly tighter LP relaxations.

    Args:
        model: The Gurobi model instance (passed automatically by Gurobi).
        where: Callback code indicating when the callback is being invoked.
    """
    if where == GRB.Callback.MIPSOL:
        # Retrieve the current integer solution values
        x_vars = model._x_vars
        num_nodes = model._num_nodes

        # Build a graph from active edges (x[i,j] > 0.5 in the current solution)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for (i, j), var in x_vars.items():
            val = model.cbGetSolution(var)
            if val > 0.5:
                G.add_edge(i, j)

        # Find all connected components
        components = list(nx.connected_components(G))

        # For each component that does NOT contain the depot, add a DFJ cut
        for component in components:
            if 0 not in component and len(component) >= 2:
                # Component is a subtour - add lazy constraint
                # sum_{i,j in S} x[i,j] <= |S| - 1
                subtour_edges = []
                for i in component:
                    for j in component:
                        if i != j and (i, j) in x_vars:
                            subtour_edges.append(x_vars[(i, j)])

                if subtour_edges:
                    model.cbLazy(quicksum(subtour_edges) <= len(component) - 1)

        # Check capacity for routes connected to the depot
        for component in components:
            _check_route_capacity(model, G, x_vars, component)


def _setup_ks_model(
    model: gp.Model,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    use_binary_vars: bool = False,
) -> Tuple[Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    r"""
    Setup the base mathematical parameters, variables, and constraints for the VRPP model.

    **CRITICAL CHANGE**: This function now uses a Branch-and-Cut architecture with
    Dantzig-Fulkerson-Johnson (DFJ) lazy constraints instead of Miller-Tucker-Zemlin (MTZ).

    Mathematical Justification:
        MTZ uses auxiliary continuous variables u[i] and Big-M constraints:
            u[j] >= u[i] + d[j] - Q(1 - x[i,j])

        The continuous LP relaxation of MTZ is notoriously weak because:
        1. The Big-M coefficient Q creates a large feasible region
        2. Fractional x[i,j] values allow u[i] to "float" arbitrarily
        3. The polyhedral bounds are loose, leading to poor dual bounds

        DFJ, in contrast, directly enforces combinatorial structure:
            sum_{i,j in S} x[i,j] <= |S| - 1  for all S ⊂ V \ {0}

        This yields a dramatically tighter LP relaxation because:
        1. Constraints are purely in the x-space (no auxiliary variables)
        2. Fractional solutions that "look like" subtours are immediately cut off
        3. The LP bound converges much faster to the integer optimum

        For instances with n > 40, MTZ typically fails to find integer solutions within
        reasonable time limits, while DFJ scales effectively to n = 100-200.

    Args:
        model (gp.Model): The Gurobi model instance.
        dist_matrix (np.ndarray): Cost matrix between all nodes (0 is depot).
        wastes (Dict[int, float]): Waste level at each node.
        capacity (float): Capacity of the vehicle.
        R (float): Revenue multiplier for collected waste.
        C (float): Cost multiplier for distance traveled.
        mandatory_nodes (List[int]): Nodes that MUST have their `y` variable set to 1.
        use_binary_vars (bool): If True, create BINARY variables directly. If False (default),
            create CONTINUOUS variables for LP relaxation (Kernel Search workflow).

    Returns:
        Tuple: A 2-tuple containing:
            - `x` (Dict[Tuple[int, int], gp.Var]): binary/continuous flow variables x_{i,j}.
            - `y` (Dict[int, gp.Var]): binary/continuous selection variables y_i.
    """
    num_nodes = len(dist_matrix)
    nodes = list(range(num_nodes))
    customers = list(range(1, num_nodes))
    m_set = set(mandatory_nodes)

    # Determine variable type based on usage context
    var_type = GRB.BINARY if use_binary_vars else GRB.CONTINUOUS

    # 1. Decision Variables
    # Flow variables: x[i,j] = 1 if the vehicle travels from i to j
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(lb=0, ub=1, vtype=var_type, name=f"x_{i}_{j}")

    # Selection variables: y[i] = 1 if customer i is visited
    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=var_type, name=f"y_{i}")

    # 2. Objective Function: Maximize (Revenue - Cost)
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # 3. Flow Balance and Selection Constraints
    for i in customers:
        # If node i is visited, exactly one edge enters and one edge leaves
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i])
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i])

    # Depot usage: at most one trip leaves the depot
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1)
    # Conservation of flow at the depot
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers))

    # 4. Capacity Constraint (Simplified - no MTZ variables needed)
    # We enforce capacity via the greedy construction and lazy cuts
    # The DFJ callback will handle subtour elimination dynamically

    # 5. Mandatory Visits
    for i in m_set:
        model.addConstr(y[i] == 1)

    # 6. Enable Lazy Constraint Mode
    # This MUST be set BEFORE calling optimize() with a callback
    model.Params.LazyConstraints = 1

    # Store metadata in model for callback access
    model._x_vars = x
    model._num_nodes = num_nodes
    model._wastes = wastes
    model._capacity = capacity

    return x, y


def _set_mip_start(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
):
    """
    Warm-start the MIP solver with a greedy heuristic solution.

    This function computes a feasible solution using the greedy construction heuristic
    and injects it into the Gurobi model via the .Start attribute. This ensures that
    the solver begins with SolCount = 1, which is critical for matheuristics like
    Local Branching that require an incumbent to define the k-neighborhood.

    Mathematical Rationale:
        Warm-starting with a feasible solution provides:
        1. An immediate upper bound (for maximization problems)
        2. A starting point for the branch-and-bound tree
        3. A baseline for local search neighborhoods (LB, LB-VNS)
        4. Improved primal-dual gap closure rates

    Args:
        model: The Gurobi model instance.
        x: Edge flow variables.
        y: Node selection variables.
        dist_matrix: Distance matrix.
        wastes: Waste dictionary.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: Nodes that must be visited.
    """
    rng = random.Random()
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist_matrix, wastes=wastes, capacity=capacity, R=R, C=C, mandatory_nodes=mandatory_nodes, rng=rng
    )

    # Initialize all variables to 0
    for var in x.values():
        var.Start = 0.0
    for var in y.values():
        var.Start = 0.0

    # Set variables based on heuristic routes
    visited_nodes = set()
    for route in heuristic_routes:
        if not route:
            continue

        # First edge: depot to first customer
        if (0, route[0]) in x:
            x[(0, route[0])].Start = 1.0

        # Interior edges
        for i in range(len(route) - 1):
            if (route[i], route[i + 1]) in x:
                x[(route[i], route[i + 1])].Start = 1.0
            visited_nodes.add(route[i])

        # Last edge: last customer to depot
        if len(route) > 0:
            if (route[-1], 0) in x:
                x[(route[-1], 0)].Start = 1.0
            visited_nodes.add(route[-1])

    # Set y variables for visited nodes
    for node in visited_nodes:
        if node in y:
            y[node].Start = 1.0


def _separate_fractional_subtours(model, x_vars, num_nodes):
    """
    Identify and inject fractional subtour elimination cuts (User Cuts).
    Uses Max-Flow/Min-Cut to find violated directed cuts.
    """
    # 1. Build a capacity graph from the fractional solution
    G = nx.DiGraph()
    for (i, j), var in x_vars.items():
        val = model.cbGetNodeRel(var)
        if val > 1e-4:
            G.add_edge(i, j, capacity=val)

    # 2. Check cuts from depot (0) to each customer i
    # Requirement: min-cut(0 -> i) >= y_i
    for i in range(1, num_nodes):
        if i not in G:
            continue

        y_val = model.cbGetNodeRel(model._y_vars[i])
        if y_val < 1e-4:
            continue

        cut_value, (reachable, _non_reachable) = nx.minimum_cut(G, 0, i)

        if cut_value < y_val - 1e-4:
            # Violated cut: sum_{u in S, v not in S} x[u,v] >= y[i]
            # S is the set containing the depot (0)
            S = reachable
            cut_edges = []
            for u in S:
                if u in G:
                    for v in G.neighbors(u):
                        if v not in S:
                            cut_edges.append(x_vars[(u, v)])

            if cut_edges:
                model.cbCut(quicksum(cut_edges) >= model._y_vars[i])


def _root_node_callback(model, where):
    """
    Gurobi callback to harvest the root node relaxation values.
    ENHANCED: Adds fractional subtour elimination user cuts before harvesting.
    """
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL and model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            # 1. Separate fractional subtours at the root node
            _separate_fractional_subtours(model, model._x_vars, model._num_nodes)

            # 2. Harvest relaxation values (now reflecting the subtour cuts)
            model._node_rel = model.cbGetNodeRel(model._all_vars_list)
            model.terminate()


def _get_partitioned_vars(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    initial_kernel_size: int,
    bucket_size: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
) -> Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
    """
    Solve the root node relaxation and partition discrete variables into Kernel and Buckets.
    Ranking is based on subtour-aware relaxation values.

    Note: A unified list is used for all variables (x and y) because this VRPP
    formulation is a pure 0-1 MIP. While Guastaroba et al. (2017) suggests
    maintaining separate lists for binary and general integer variables, they
    are merged here as all discrete variables in this model are binary.
    """
    # Ensure Lazy Constraints are enabled for any solve with callbacks
    model.Params.LazyConstraints = 1

    # Prepare for root node relaxation harvesting
    all_vars_list = list(x.values()) + list(y.values())
    model._all_vars_list = all_vars_list
    model._node_rel = [0.0] * len(all_vars_list)
    model._y_vars = y  # Needed for separation logic

    # variables already binary from _setup_ks_model(use_binary_vars=True) logic in run call

    # Optimize with the root node callback
    model.optimize(_root_node_callback)

    # 3. Extract relaxation values
    var_values = {var: val for var, val in zip(all_vars_list, model._node_rel, strict=False)}

    # 4. Compute a totally feasible heuristic route for robustness
    rng = random.Random()
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist_matrix, wastes=wastes, capacity=capacity, R=R, C=C, mandatory_nodes=mandatory_nodes, rng=rng
    )

    heuristic_edges = set()
    heuristic_nodes = set()
    for route in heuristic_routes:
        if not route:
            continue
        heuristic_edges.add((0, route[0]))
        for i in range(len(route) - 1):
            heuristic_edges.add((route[i], route[i + 1]))
            heuristic_nodes.add(route[i])
        heuristic_edges.add((route[-1], 0))
        heuristic_nodes.add(route[-1])

    # 5. Extract variables and their harvested relaxation values
    all_vars = []
    for (i, j), var in x.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "x", (i, j)))
    for i, var in y.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "y", i))  # type: ignore[arg-type]

    # Sort logic: Root node relaxation value descending
    all_vars.sort(key=lambda item: item[1], reverse=True)

    # 6. Partition into sets
    lp_support_size = sum(1 for _, val, _, _ in all_vars if val > 1e-4)
    actual_kernel_size = max(initial_kernel_size, lp_support_size)

    kernel_vars = []
    remaining_vars = []

    # Include heuristic variables to guarantee ILP feasibility
    for v_obj, _, _, _ in all_vars[:actual_kernel_size]:
        kernel_vars.append(v_obj)

    for v_obj, _, vtype, idx in all_vars[actual_kernel_size:]:
        if (vtype == "x" and idx in heuristic_edges) or (vtype == "y" and idx in heuristic_nodes):
            kernel_vars.append(v_obj)
        else:
            remaining_vars.append(v_obj)

    buckets = [remaining_vars[i : i + bucket_size] for i in range(0, len(remaining_vars), bucket_size)]

    return kernel_vars, remaining_vars, buckets


def _solve_ks_iterations(
    model: gp.Model,
    kernel_vars: List[gp.Var],
    buckets: List[List[gp.Var]],
    remaining_vars: List[gp.Var],
    time_limit: float,
    mip_limit_nodes: int,
) -> Set[gp.Var]:
    """
    Perform the iterative improvement phase by solving sub-problems with DFJ cuts.

    Each iteration adds one bucket of variables and a budget constraint.
    """
    # Fix everything to zero initially
    for v in remaining_vars:
        v.UB = 0

    best_obj = -float("inf")
    used_vars = set()
    start_time = model.Runtime

    # 1. Solve INITIAL KERNEL with DFJ callback
    model.setParam("TimeLimit", max(2.0, (time_limit / (len(buckets) + 1))))
    model.setParam("NodeLimit", mip_limit_nodes)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount > 0:
        best_obj = model.ObjVal
        used_vars = {v for v in kernel_vars if v.X > 0.5}

    # 2. Iterative BUCKET refinement
    for bucket in buckets:
        # Check time budget
        if (model.Runtime - start_time) > time_limit:
            break

        # Dynamically unfix the current bucket
        for v in bucket:
            v.UB = 1

        # KS Bucket Constraint: sum_{j in B_i} x_j >= 1
        bucket_constr = model.addConstr(quicksum(bucket) >= 1, name="KS_Bucket_Constraint")

        # Step 2: Objective Cutoff (Guastaroba 2017, Step 3.1)
        # Solutions worse than this will be discarded.
        model.Params.Cutoff = best_obj + 1e-4

        model.optimize(_dfj_subtour_elimination_callback)

        if model.SolCount > 0:
            if model.ObjVal > best_obj:
                best_obj = model.ObjVal
                # If improvement, capture NEW useful variables
                for v in bucket:
                    if v.X > 0.5:
                        used_vars.add(v)

            # Fix unused variables in the bucket back to zero to keep sub-MIP small
            for v in bucket:
                if v.X < 0.5:
                    v.UB = 0
        else:
            # Revert unfixing if no better solution found
            for v in bucket:
                v.UB = 0

        # RIGOR: Explicitly remove the bucket constraint after the solve
        model.remove(bucket_constr)

    return used_vars


def _reconstruct_tour(
    num_nodes: int, x: Dict[Tuple[int, int], Any], dist_matrix: np.ndarray
) -> Tuple[List[int], float]:
    """
    Convert the resulting binary flow variables into a sequence of nodes (a tour).

    Args:
        num_nodes: Total number of nodes in graph.
        x: Flow variables from the Gurobi model (either gp.Var with .X or floats).
        dist_matrix: Matrix used for distance calculation.

    Returns:
        Tuple: (tour list, total routing cost)
    """
    # 1. Extract activated edges
    active_edges = []
    for (i, j), var in x.items():
        val = getattr(var, "X", var)
        if val > 0.5:
            active_edges.append((i, j))

    if not active_edges:
        return [0, 0], 0.0

    # 2. Build adjacency map for path reconstruction
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for i, j in active_edges:
        adj[i].append(j)

    # 3. Traverse the path starting at depot 0
    full_tour = [0]
    current = 0
    visited_nodes = {0}
    while True:
        if not adj[current]:
            break
        nx_node = adj[current].pop(0)
        # Handle potential cycle/convergence to zero
        if nx_node in visited_nodes and nx_node != 0:
            break
        full_tour.append(nx_node)
        visited_nodes.add(nx_node)
        current = nx_node
        if current == 0:
            break

    # 4. Ensure it closes at the depot
    if full_tour[-1] != 0:
        full_tour.append(0)

    # 5. Calculate final cost
    routing_cost = sum(dist_matrix[i][j] for i, j in active_edges)
    return full_tour, float(routing_cost)


def run_kernel_search_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    initial_kernel_size: int = 50,
    bucket_size: int = 20,
    max_buckets: int = 10,
    time_limit: float = 300.0,
    mip_limit_nodes: int = 5000,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Entry point to solve the VRPP via Kernel Search matheuristic.

    Orchestrates the model setup, variable partitioning, iterative improvement,
    and final tour reconstruction.

    Args:
        dist_matrix: Pairwise distance matrix.
        wastes: Node demands/waste levels.
        capacity: Vehicle capacity.
        R: Revenue multiplier for collected waste.
        C: Cost multiplier for distance.
        mandatory_nodes: Nodes that must be visited.
        initial_kernel_size: Number of variables in the primary search space.
        bucket_size: Number of variables explored per iteration.
        max_buckets: Maximum iterations (bucket solves).
        time_limit: Total wall-clock runtime limit (seconds).
        mip_limit_nodes: Maximum branch-and-bound nodes per sub-MIP solve.
        mip_gap: Targeted optimality gap for each sub-problem.
        seed: Random seed for solver reproducibility.
        env: Optional pre-configured Gurobi environment.
        recorder: Optional utility for tracking search progress/statistics.

    Returns:
        Tour, MILP Objective value, and physical Routing Cost.
    """
    # 1. Initialize Model
    model = gp.Model("KS_VRPP", env=env) if env else gp.Model("KS_VRPP")
    model.setParam("OutputFlag", 0)  # Deactivate verbose output
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # Phase 1: Structural Setup (ensure binary vars for root relaxation)
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=True)

    # Phase 2: LP Relaxation & Variable Partitioning
    # This phase creates the ranking of variables for the Kernel and Buckets.
    kernel_vars, remaining_vars, buckets = _get_partitioned_vars(
        model, x, y, initial_kernel_size, bucket_size, dist_matrix, wastes, capacity, R, C, mandatory_nodes
    )
    if not kernel_vars:
        # Fallback if LP solve failed to find even partial feasibility
        return [0, 0], 0.0, 0.0

    # Limit search space by max_buckets parameter
    buckets = buckets[:max_buckets]

    # Phase 3: Iterative Search (Incremental MIPs)
    # The kernel grows and unused variables are pruned in each step.
    used_vars = _solve_ks_iterations(model, kernel_vars, buckets, remaining_vars, time_limit, mip_limit_nodes)

    # Phase 4: Final extraction and tour reconstruction
    if not used_vars:
        return [0, 0], 0.0, 0.0

    # Unfix all variables identified as useful for a final "polishing" solve
    for v in used_vars:
        v.UB = 1

    # Warm-start with greedy heuristic to ensure feasibility
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)

    # 5. Analytics tracking
    if recorder:
        recorder.record(engine="kernel_search", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
