"""
Fix-and-Optimize (Sub-MIP) Intensification Operator.

Selects a subset of routes to optimise exactly while holding the remainder
fixed.  The "free" routes' customers are extracted and handed to Gurobi as a
small CVRP or VRPP sub-MIP.  The proven-optimal recombination found by the
solver replaces the original free routes in the plan.

Selection strategy:
    Routes are ranked by quality (total distance / total load for CVRP, or
    profit for VRPP).  The ``free_fraction`` worst-quality routes are
    designated "free" and submitted to the sub-MIP.  All other routes remain
    fixed and are appended unchanged to the final plan.

Sub-MIP formulation (CVRP — Miller-Tucker-Zemlin):
    Variables  x[i,j] ∈ {0,1}  (arc traversal), u[i] ∈ [demand_i, Q]  (MTZ load).
    Objective  Minimize Σ d[i,j] · x[i,j].
    Constraints:
        Σ_j x[i,j] = 1  ∀ customer i      (depart each customer once)
        Σ_i x[i,j] = 1  ∀ customer j      (enter each customer once)
        Σ_j x[0,j] = K_free               (exactly K_free vehicles from depot)
        u[j] ≥ u[i] + demand[j] − Q(1 − x[i,j])   ∀ i,j ∈ customers  (MTZ)

Sub-MIP formulation (VRPP):
    Additional binary y[i] ∈ {0,1} (whether customer i is visited).
    Objective  Maximize Σ waste[i]·R·y[i] − Σ d[i,j]·C·x[i,j].
    Flow conservation uses y[i] instead of 1:  Σ_j x[i,j] = y[i].
    MTZ:  u[j] ≥ u[i] + demand[j] − Q(1 − x[i,j])  (same as CVRP).

If Gurobi finds no feasible integer solution within *time_limit* the original
free routes are returned unchanged as a fallback.

Requirements:
    Gurobi Optimizer ≥ 11.0 with a valid licence.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import fix_and_optimize
    >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity)
    >>> improved = fix_and_optimize_profit(routes, dist_matrix, wastes, capacity,
    ...                                    R=1.0, C=0.5)
"""

from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Route quality scorers
# ---------------------------------------------------------------------------


def _route_distance(route: List[int], dist_matrix: np.ndarray) -> float:
    """Total travel distance of *route* (depot-inclusive round trip)."""
    if not route:
        return 0.0
    cost = float(dist_matrix[0, route[0]])
    for k in range(len(route) - 1):
        cost += float(dist_matrix[route[k], route[k + 1]])
    cost += float(dist_matrix[route[-1], 0])
    return cost


def _route_profit(
    route: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> float:
    """Net profit of *route*: revenue − C · distance."""
    revenue = sum(wastes.get(n, 0.0) for n in route) * R
    return revenue - _route_distance(route, dist_matrix) * C


# ---------------------------------------------------------------------------
# Gurobi CVRP sub-MIP solver
# ---------------------------------------------------------------------------


def _solve_cvrp_mip(  # noqa: C901
    free_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    n_vehicles: int,
    time_limit: float,
    seed: int,
) -> List[List[int]]:
    """Solve a CVRP sub-MIP for *free_nodes* using Gurobi (MTZ formulation).

    All nodes in *free_nodes* are mandatory.  The solver minimises total
    travel distance with at most *n_vehicles* vehicles.

    Args:
        free_nodes: Customer node IDs to optimise (actual dist_matrix indices).
        dist_matrix: Full distance matrix; depot at index 0.
        wastes: Node demand lookup.
        capacity: Vehicle capacity Q.
        n_vehicles: Maximum number of vehicles to use (K_free).
        time_limit: Gurobi wall-clock time limit in seconds.
        seed: Gurobi random seed.

    Returns:
        List of routes found by the sub-MIP, or one route per customer as a
        fallback if no integer solution is found within the time limit.
    """
    N = len(free_nodes)
    if N == 0:
        return []

    # verts[0] = depot (0), verts[1..N] = customers (free_nodes)
    verts = [0] + free_nodes

    m = gp.Model("fix_and_opt_cvrp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Seed", seed)
    m.setParam("MIPGap", 1e-4)

    # Arc variables
    x: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # MTZ load variables u[1..N] ∈ [demand_i, Q]
    u: Dict[int, gp.Var] = {}
    for i in range(1, N + 1):
        dem = wastes.get(verts[i], 0.0)
        u[i] = m.addVar(lb=dem, ub=capacity, name=f"u_{i}")

    m.update()

    # Objective: minimise total arc cost
    m.setObjective(
        gp.quicksum(dist_matrix[verts[i], verts[j]] * x[i, j] for i in range(N + 1) for j in range(N + 1) if i != j),
        GRB.MINIMIZE,
    )

    # Each customer entered exactly once
    for j in range(1, N + 1):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N + 1) if i != j) == 1)

    # Each customer departed exactly once
    for i in range(1, N + 1):
        m.addConstr(gp.quicksum(x[i, j] for j in range(N + 1) if i != j) == 1)

    # At most n_vehicles vehicles from depot
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N + 1)) <= n_vehicles)

    # Flow balance at depot
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N + 1)) == gp.quicksum(x[i, 0] for i in range(1, N + 1)))

    # MTZ subtour-elimination constraints
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                dem_j = wastes.get(verts[j], 0.0)
                m.addConstr(u[j] >= u[i] + dem_j - capacity * (1 - x[i, j]))

    m.optimize()

    # Extract routes from solution
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        # Fallback: one customer per route (always feasible)
        return [[n] for n in free_nodes]

    routes_out: List[List[int]] = []
    visited: Set[int] = set()

    for start in range(1, N + 1):
        if x[0, start].X < 0.5:
            continue
        if verts[start] in visited:
            continue

        route: List[int] = []
        curr = start
        max_steps = N + 1

        while curr != 0 and max_steps > 0:
            route.append(verts[curr])
            visited.add(verts[curr])
            moved = False
            for nxt in range(N + 1):
                if nxt != curr and x[curr, nxt].X > 0.5:
                    curr = nxt
                    moved = True
                    break
            if not moved:
                break
            max_steps -= 1

        if route:
            routes_out.append(route)

    # Ensure every free node is covered (safety net for numerical edge-cases)
    covered: Set[int] = {n for r in routes_out for n in r}
    for node in free_nodes:
        if node not in covered:
            routes_out.append([node])

    return routes_out


# ---------------------------------------------------------------------------
# Gurobi VRPP sub-MIP solver
# ---------------------------------------------------------------------------


def _solve_vrpp_mip(  # noqa: C901
    free_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_vehicles: int,
    time_limit: float,
    seed: int,
    mandatory_nodes: Optional[Set[int]] = None,
) -> List[List[int]]:
    """Solve a VRPP sub-MIP for *free_nodes* using Gurobi.

    Customers in *free_nodes* are optional unless they appear in
    *mandatory_nodes*.  The solver maximises:
        profit = Σ waste[i]·R·y[i] − Σ d[i,j]·C·x[i,j]

    Args:
        free_nodes: Customer node IDs (actual dist_matrix indices).
        dist_matrix: Full distance matrix; depot at index 0.
        wastes: Node demand lookup.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        n_vehicles: Maximum number of vehicles.
        time_limit: Gurobi time limit in seconds.
        seed: Gurobi random seed.
        mandatory_nodes: Nodes that must be visited regardless of profit.

    Returns:
        List of profitable routes found by the sub-MIP.
    """
    N = len(free_nodes)
    if N == 0:
        return []

    mandatory: Set[int] = mandatory_nodes or set()
    verts = [0] + free_nodes  # verts[0] = depot

    m = gp.Model("fix_and_opt_vrpp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Seed", seed)
    m.setParam("MIPGap", 1e-4)

    # y[i] = 1 if customer verts[i] is visited, i in 1..N
    y: Dict[int, gp.Var] = {i: m.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(1, N + 1)}

    # Arc variables x[i,j]
    x: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # MTZ load variables u[1..N]
    u: Dict[int, gp.Var] = {i: m.addVar(lb=0.0, ub=capacity, name=f"u_{i}") for i in range(1, N + 1)}

    m.update()

    # Objective: maximise profit (implemented as minimise negative profit)
    revenue_expr = gp.quicksum(wastes.get(verts[i], 0.0) * R * y[i] for i in range(1, N + 1))
    cost_expr = gp.quicksum(
        dist_matrix[verts[i], verts[j]] * C * x[i, j] for i in range(N + 1) for j in range(N + 1) if i != j
    )
    m.setObjective(revenue_expr - cost_expr, GRB.MAXIMIZE)

    # Flow conservation: if visited, entered and departed exactly once
    for i in range(1, N + 1):
        m.addConstr(gp.quicksum(x[i, j] for j in range(N + 1) if j != i) == y[i])
        m.addConstr(gp.quicksum(x[j, i] for j in range(N + 1) if j != i) == y[i])

    # At most n_vehicles vehicles
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N + 1)) <= n_vehicles)

    # Mandatory nodes must be visited
    for i in range(1, N + 1):
        if verts[i] in mandatory:
            m.addConstr(y[i] == 1)

    # MTZ subtour elimination (also enforces capacity)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                dem_j = wastes.get(verts[j], 0.0)
                m.addConstr(u[j] >= u[i] + dem_j - capacity * (1 - x[i, j]))

    # Demand lower bound when visited
    for i in range(1, N + 1):
        dem_i = wastes.get(verts[i], 0.0)
        m.addConstr(u[i] >= dem_i * y[i])

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return []

    # Extract routes
    routes_out: List[List[int]] = []
    visited: Set[int] = set()

    for start in range(1, N + 1):
        if x[0, start].X < 0.5:
            continue
        if verts[start] in visited:
            continue

        route: List[int] = []
        curr = start
        max_steps = N + 1

        while curr != 0 and max_steps > 0:
            route.append(verts[curr])
            visited.add(verts[curr])
            moved = False
            for nxt in range(N + 1):
                if nxt != curr and x[curr, nxt].X > 0.5:
                    curr = nxt
                    moved = True
                    break
            if not moved:
                break
            max_steps -= 1

        if route:
            routes_out.append(route)

    # Ensure mandatory nodes appear even if solver dropped them
    covered: Set[int] = {n for r in routes_out for n in r}
    for node in free_nodes:
        if node in mandatory and node not in covered:
            routes_out.append([node])

    return routes_out


# ---------------------------------------------------------------------------
# Public API — CVRP
# ---------------------------------------------------------------------------


def fix_and_optimize(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    n_free: Optional[int] = None,
    free_fraction: float = 0.30,
    time_limit: float = 30.0,
    seed: int = 42,
) -> List[List[int]]:
    """Fix a subset of routes and solve a CVRP Sub-MIP for the remainder.

    Designates the worst-quality routes as "free" (scored by total distance ÷
    total load) and passes their customers to Gurobi as a small CVRP sub-MIP.
    All remaining "fixed" routes are appended unchanged to the result.

    The number of free routes is ``max(1, int(K · free_fraction))``, where K
    is the total number of routes, unless *n_free* is specified explicitly.

    If Gurobi finds no integer solution within *time_limit*, the original free
    routes are returned as-is (graceful fallback).

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix ``(N+1, N+1)``, depot at index 0.
        wastes: Mapping from node index to demand.
        capacity: Maximum vehicle load.
        n_free: Exact number of routes to designate as free.  If ``None``
            (default), ``free_fraction`` is used.
        free_fraction: Fraction of routes to optimise exactly.  The
            ``free_fraction`` worst-quality routes are selected.  Default 0.30.
        time_limit: Gurobi time limit in seconds.  Larger values yield better
            solutions at the cost of longer wall-clock time.  Default 30.
        seed: Gurobi random seed for reproducibility.

    Returns:
        List[List[int]]: Improved routes.  All free-route customers are present
            in the returned plan (mandatory coverage).  Fixed routes are
            returned unchanged.  The input *routes* is never mutated.

    Example:
        >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity)
        >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity,
        ...                             free_fraction=0.5, time_limit=60.0)
    """
    if not routes:
        return []

    K = len(routes)

    # Score each route by distance-per-unit-load (higher = worse quality)
    scores: List[Tuple[float, int]] = []
    for i, route in enumerate(routes):
        dist = _route_distance(route, dist_matrix)
        load = sum(wastes.get(n, 0.0) for n in route)
        scores.append((dist / max(load, 1e-9), i))

    scores.sort(reverse=True)  # Worst quality first

    k_free = n_free if n_free is not None else max(1, int(K * free_fraction))
    k_free = min(k_free, K)

    free_indices: Set[int] = {idx for _, idx in scores[:k_free]}
    fixed_routes = [routes[i] for i in range(K) if i not in free_indices]
    free_routes = [routes[i] for i in range(K) if i in free_indices]
    free_nodes = [n for r in free_routes for n in r]

    if not free_nodes:
        return [list(r) for r in routes]

    optimal_free = _solve_cvrp_mip(
        free_nodes,
        dist_matrix,
        wastes,
        capacity,
        n_vehicles=k_free,
        time_limit=time_limit,
        seed=seed,
    )

    return [list(r) for r in fixed_routes] + optimal_free


# ---------------------------------------------------------------------------
# Public API — VRPP
# ---------------------------------------------------------------------------


def fix_and_optimize_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_free: Optional[int] = None,
    free_fraction: float = 0.30,
    time_limit: float = 30.0,
    seed: int = 42,
    mandatory_nodes: Optional[List[int]] = None,
) -> List[List[int]]:
    """VRPP Fix-and-Optimize: maximise profit over the free-route Sub-MIP.

    Selects the lowest-profit routes as "free" (scored by net profit:
    revenue − C · distance), hands their customers to a Gurobi VRPP sub-MIP
    (customers optional, depot fixed), and replaces the free routes with the
    profit-maximising solution found.

    Nodes in *mandatory_nodes* are forced into the sub-MIP solution even if
    their inclusion is unprofitable.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        n_free: Exact number of routes to optimise.  Overrides *free_fraction*.
        free_fraction: Fraction of routes to optimise.  Default 0.30.
        time_limit: Gurobi time limit in seconds.  Default 30.
        seed: Gurobi random seed.
        mandatory_nodes: Nodes that must appear in the final solution.

    Returns:
        List[List[int]]: Improved routes.  Fixed routes are unchanged; free
            routes are replaced by the Gurobi profit-maximising solution.
            The input *routes* is never mutated.

    Example:
        >>> improved = fix_and_optimize_profit(routes, dist_matrix, wastes,
        ...                                    capacity, R=1.0, C=0.5)
    """
    if not routes:
        return []

    K = len(routes)
    mandatory_set: Set[int] = set(mandatory_nodes) if mandatory_nodes else set()

    # Score each route by profit (lower profit = worse quality → free first)
    scores: List[Tuple[float, int]] = []
    for i, route in enumerate(routes):
        profit = _route_profit(route, dist_matrix, wastes, R, C)
        scores.append((profit, i))

    scores.sort()  # Lowest profit first (worst quality)

    k_free = n_free if n_free is not None else max(1, int(K * free_fraction))
    k_free = min(k_free, K)

    free_indices: Set[int] = {idx for _, idx in scores[:k_free]}
    fixed_routes = [routes[i] for i in range(K) if i not in free_indices]
    free_routes = [routes[i] for i in range(K) if i in free_indices]
    free_nodes = [n for r in free_routes for n in r]

    if not free_nodes:
        return [list(r) for r in routes]

    optimal_free = _solve_vrpp_mip(
        free_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        n_vehicles=k_free,
        time_limit=time_limit,
        seed=seed,
        mandatory_nodes=mandatory_set,
    )

    return [list(r) for r in fixed_routes] + optimal_free
