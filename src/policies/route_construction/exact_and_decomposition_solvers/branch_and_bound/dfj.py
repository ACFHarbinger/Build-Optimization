"""
DFJ-formulation Branch-and-Bound solver using Gurobi's built-in B&B engine.

Uses the Dantzig-Fulkerson-Johnson (DFJ) formulation with lazy subtour elimination
constraints. This delegates the B&B tree search to Gurobi's highly optimized internal
engine, adding subtour cuts dynamically during the search.

Reference:
    Dantzig, G., Fulkerson, R., & Johnson, S. (1954).
    "Solution of a large-scale traveling-salesman problem".
    Journal of Operations Research, 2(4), 393-410.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .params import BBParams


def _dfj_callback(model, where):
    """DFJ subtour elimination callback for Gurobi's built-in B&B."""
    if where == GRB.Callback.MIPSOL:
        x_vars = model._x_vars
        num_nodes = model._num_nodes

        # Build graph from active edges
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for (i, j), var in x_vars.items():
            val = model.cbGetSolution(var)
            if val > 0.5:
                G.add_edge(i, j)

        # Find connected components
        components = list(nx.connected_components(G))

        # Add cuts for subtours
        for component in components:
            if 0 not in component and len(component) >= 2:
                # For directed formulation, cut outgoing flow from component to rest.
                # Each visited node in the component must have at least one edge
                # leaving the component: sum_{i in S, j not in S} x[i,j] >= y[i] for i in S.
                # Simpler valid cut: sum of outgoing edges from S >= 1
                # (at least one edge must leave S to connect back to depot).
                rest = set(range(num_nodes)) - component
                outgoing_edges = [x_vars[i, j] for i in component for j in rest if (i, j) in x_vars]
                if outgoing_edges:
                    model.cbLazy(quicksum(outgoing_edges) >= 1)


def _setup_bb_model(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    mandatory_indices: Set[int],
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    """Set up the Gurobi model for VRPP."""
    num_nodes = len(dist_matrix)
    nodes = range(num_nodes)
    customers = list(range(1, num_nodes))

    model = gp.Model("BB_Simple", env=env) if env else gp.Model("BB_Simple")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Seed", seed)
    model.Params.LazyConstraints = 1

    # Decision variables (BINARY)
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"y_{i}")

    # Objective and constraints set in run_bb_simple to keep params local if needed,
    # but we can pass R and C here or just return the variables.
    return model, x, y


def _extract_routes_from_adj(adj: Dict[int, List[int]], num_nodes: int) -> List[List[int]]:
    """Extract routes starting from the depot using an adjacency list."""
    routes = []
    for start in adj[0]:
        route = [start]
        current = start
        visited = {0, start}

        while current != 0 and adj[current]:
            next_node = adj[current][0]
            if next_node in visited and next_node != 0:
                break
            if next_node != 0:
                route.append(next_node)
                visited.add(next_node)
            current = next_node
            if current == 0:
                break

        if route:
            routes.append(route)
    return routes


def run_bb_dfj(
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
    Dispatcher entry point for the DFJ-formulation Branch-and-Bound solver.

    Uses Dantzig-Fulkerson-Johnson (DFJ) formulation with lazy subtour elimination
    constraints. Delegates the branch-and-bound tree search to Gurobi's highly
    optimized internal engine.

    Args:
        dist_matrix: Symmetric distance matrix (n x n).
        wastes: Mapping of customer IDs to fill levels/profits.
        capacity: Vehicle payload capacity.
        R: Revenue coefficient per unit collected.
        C: Cost coefficient per unit distance.
        values: Configuration dictionary (time_limit, mip_gap, etc.).
        mandatory_indices: Set of mandatory customer nodes.
        env: Optional Gurobi environment for resource management.
        seed: Optional random seed.
        recorder: Optional telemetry recorder.

    Returns:
        Tuple of (routes, objective_value).
    """
    num_nodes = len(dist_matrix)
    nodes = range(num_nodes)
    customers = list(range(1, num_nodes))
    mandatory_indices = mandatory_indices or set()

    # Extract configuration
    if params is None:
        params = BBParams()

    # Setup basic model structure
    model, x, y = _setup_bb_model(
        dist_matrix=dist_matrix,
        wastes=wastes,
        mandatory_indices=mandatory_indices,
        time_limit=params.time_limit,
        mip_gap=params.mip_gap,
        seed=params.seed,
        env=env,
    )

    # Objective: Maximize (Revenue - Cost)
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # Flow balance constraints
    for i in customers:
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"out_{i}")
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"in_{i}")

    # Depot constraints
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1, name="depot_out")
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers), name="depot_balance")

    # Mandatory nodes
    for i in mandatory_indices:
        if i in y:
            model.addConstr(y[i] == 1, name=f"mandatory_{i}")

    # Capacity constraint
    # The total waste collected on any route must not exceed vehicle capacity
    model.addConstr(quicksum(wastes.get(i, 0) * y[i] for i in customers) <= capacity, name="capacity")

    # Store for callback
    model._x_vars = x
    model._num_nodes = num_nodes

    # Solve with DFJ callback
    model.optimize(_dfj_callback)

    # Extract solution
    if model.SolCount == 0:
        return [], 0.0

    active_edges = [(i, j) for (i, j), var in x.items() if var.X > 0.5]
    if not active_edges:
        return [], 0.0

    # Build adjacency list and extract routes
    adj: dict[int, list[int]] = {i: [] for i in nodes}
    for i, j in active_edges:
        adj[i].append(j)

    routes = _extract_routes_from_adj(adj, num_nodes)

    # Record stats
    if recorder:
        recorder.record(
            engine="bb_dfj",
            obj_val=model.ObjVal,
            nodes_explored=int(model.NodeCount),
            time=model.Runtime,
            solved=1,
        )

    return routes, float(model.ObjVal)
