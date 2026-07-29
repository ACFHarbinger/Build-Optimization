"""
Wrapper for Google OR-Tools to solve the waste-collecting VRP.

This module adapts the WSmart-Route problem definition into a format
compatible with OR-Tools' routing solver.
"""

import numpy as np
from google.protobuf.duration_pb2 import Duration
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def run_alns_ortools(dist_matrix, wastes, capacity, R, C, values):
    """
    Run ALNS using Google OR-Tools with Guided Local Search.
    """
    seed = values.get("seed", 42)

    # 1. Identify active nodes (Depot + keys in wastes)
    active_nodes = [0] + sorted(list(wastes.keys()))
    compact_to_global = {c: g for c, g in enumerate(active_nodes)}

    n_active = len(active_nodes)
    max_vehicles = n_active

    # 2. Create Distance Matrix for Compact Indices
    sub_matrix = np.zeros((n_active, n_active), dtype=int)
    for c1 in range(n_active):
        for c2 in range(n_active):
            g1 = compact_to_global[c1]
            g2 = compact_to_global[c2]
            sub_matrix[c1][c2] = int(dist_matrix[g1][g2])

    # 3. Create wastes Array
    sub_wastes = [0] * n_active
    for c in range(1, n_active):
        g = compact_to_global[c]
        sub_wastes[c] = int(wastes[g])

    scale_factor = 1000
    scaled_capacity = int(capacity * scale_factor)
    scaled_wastes = [int(d * scale_factor) for d in sub_wastes]

    # 4. OR-Tools Setup
    manager = pywrapcp.RoutingIndexManager(n_active, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return sub_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def waste_callback(from_index):
        """Returns the waste of the node."""
        from_node = manager.IndexToNode(from_index)
        return scaled_wastes[from_node]

    waste_callback_index = routing.RegisterUnaryTransitCallback(waste_callback)
    routing.AddDimensionWithVehicleCapacity(
        waste_callback_index,
        0,  # null capacity slack
        [scaled_capacity] * max_vehicles,
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    time_limit_sec = values.get("time_limit", 10)
    duration = Duration()
    duration.FromSeconds(int(time_limit_sec))
    search_parameters.time_limit.CopyFrom(duration)
    routing.solver().ReSeed(seed)

    solution = routing.SolveWithParameters(search_parameters)

    routes = []
    total_cost = 0
    profit = 0.0
    if solution:
        for vehicle_id in range(max_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            index = solution.Value(routing.NextVar(index))
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                global_idx = compact_to_global[node_index]
                route.append(global_idx)
                index = solution.Value(routing.NextVar(index))

            if route:
                routes.append(route)

        total_cost = solution.ObjectiveValue() * C
        collected_revenue = sum(wastes.get(node_idx, 0) * R for route in routes for node_idx in route)
        profit = collected_revenue - total_cost

    return routes, profit, total_cost
