"""
OR-Tools engine for Branch-Cut-and-Price module.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def run_bcp_ortools(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Waste-Collecting CVRP using Google OR-Tools.
    """
    num_nodes = len(dist_matrix)
    num_vehicles = values.get("num_vehicles", num_nodes)
    depot_index = 0

    # 1. Create Routing Manager and Model
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # 2. Add Constraints and Penalties
    _add_distance_constraints(routing, manager, dist_matrix)
    _add_capacity_constraints(routing, manager, wastes, capacity, num_vehicles)
    _add_waste_collecting_penalties(routing, manager, wastes, mandatory_nodes, R, num_nodes)

    # 3. Solve
    search_parameters = _get_search_parameters(values)
    solution = routing.SolveWithParameters(search_parameters)

    # 4. Parse Result
    if solution:
        routes = _parse_routes(routing, manager, solution, num_vehicles)
        real_dist_cost = _calculate_real_cost(routes, dist_matrix)

        if recorder is not None:
            recorder.record(engine="ortools", n_routes=len(routes), cost=real_dist_cost * C, solved=1)
        return routes, real_dist_cost * C

    if recorder is not None:
        recorder.record(engine="ortools", n_routes=0, cost=0.0, solved=0)
    return [], 0.0


def _add_distance_constraints(
    routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager, dist_matrix: np.ndarray
) -> None:
    """Define Cost distance callback."""
    SCALE = 1000

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * SCALE)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


def _add_capacity_constraints(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    wastes: Dict[int, float],
    capacity: float,
    num_vehicles: int,
) -> None:
    """Add Capacity Constraints."""

    def waste_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        return int(wastes.get(from_node, 0))

    waste_callback_index = routing.RegisterUnaryTransitCallback(waste_callback)
    routing.AddDimensionWithVehicleCapacity(waste_callback_index, 0, [int(capacity)] * num_vehicles, True, "Capacity")


def _add_waste_collecting_penalties(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    wastes: Dict[int, float],
    mandatory_nodes: Optional[List[int]],
    R: float,
    num_nodes: int,
) -> None:
    """Add Penalties (Waste Collecting)."""
    SCALE = 1000
    MUST_GO_PENALTY = 1_000_000_000
    m_set = set(mandatory_nodes) if mandatory_nodes else set()

    for i in range(1, num_nodes):
        d = wastes.get(i, 0)
        revenue = d * R
        penalty = MUST_GO_PENALTY if i in m_set else int(revenue * SCALE)
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)


def _get_search_parameters(values: Dict[str, Any]) -> Any:
    """Configure search parameters."""
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    time_limit_sec = values.get("time_limit", 30)
    search_parameters.time_limit.seconds = int(time_limit_sec)
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    return search_parameters


def _parse_routes(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    solution: pywrapcp.Assignment,
    num_vehicles: int,
) -> List[List[int]]:
    """Parse routes from the solution."""
    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        if routing.IsEnd(index):
            continue

        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue

        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                route.append(node_index)
            index = solution.Value(routing.NextVar(index))

        if route:
            routes.append(route)
    return routes


def _calculate_real_cost(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """Calculate the real distance cost of the routes."""
    real_dist_cost = 0.0
    for r in routes:
        full_r = [0] + r + [0]
        for i in range(len(full_r) - 1):
            u, v = full_r[i], full_r[i + 1]
            real_dist_cost += float(dist_matrix[u][v])
    return real_dist_cost
