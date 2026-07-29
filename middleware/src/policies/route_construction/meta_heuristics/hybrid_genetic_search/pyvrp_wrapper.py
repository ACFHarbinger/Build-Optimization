"""
Wrapper for the external `PyVRP` library.

This module provides an interface to solve the routing problem using
the high-performance PyVRP solver.
"""

import numpy as np
import pyvrp
from pyvrp.stop import MaxRuntime


def solve_pyvrp(dist_matrix, wastes, capacity, R, C, values):
    """
    Solve using external PyVRP library, matching WSmart-Route style.
    """
    # 1. Map active nodes
    active_nodes = [0] + sorted(list(wastes.keys()))
    compact_to_global = {c: g for c, g in enumerate(active_nodes)}
    n_active = len(active_nodes)

    # 2. Build Components
    clients_list = []
    # Index 0 is depot in active_nodes, so we start from 1
    for i in range(1, n_active):
        g = compact_to_global[i]
        d = int(wastes.get(g, 0))
        # Use delivery for waste as in cvrp.py
        clients_list.append(pyvrp.Client(x=0, y=0, delivery=d))

    depots_list = [pyvrp.Depot(x=0, y=0)]

    # n_vehicles=0 means unlimited -> set to n_active
    max_v = values.get("max_vehicles", n_active)
    if max_v == 0:
        max_v = n_active

    vehicle_types_list = [pyvrp.VehicleType(capacity=int(capacity), num_available=max_v)]

    # 3. Create ProblemData
    # PyVRP expects distance_matrices as a list of numpy arrays
    sub_matrix = dist_matrix[np.ix_(active_nodes, active_nodes)].astype(int)

    data = pyvrp.ProblemData(
        clients=clients_list,
        depots=depots_list,
        vehicle_types=vehicle_types_list,
        distance_matrices=[sub_matrix],
        duration_matrices=[np.zeros((n_active, n_active), dtype=int)],
    )

    # 4. Solve
    time_limit = float(values.get("time_limit", 10.0))
    res = pyvrp.solve(data, stop=MaxRuntime(time_limit), seed=values.get("seed", 42))

    # 5. Parse Result
    routes = []
    for route in res.best.routes():
        # route stores client indices (1-based relative to ProblemData.clients)
        r = [compact_to_global[node_idx] for node_idx in route]
        routes.append(r)

    total_cost = res.cost() * C
    collected_revenue = sum(wastes[n] * R for r in routes for n in r)
    profit = collected_revenue - total_cost

    return routes, profit, total_cost
