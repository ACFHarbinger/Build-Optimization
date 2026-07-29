from typing import List, Tuple

import numpy as np

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False


class RoutingSubproblem:
    """
    Subproblem solver for LBBD using OR-Tools Routing library.
    Solves the TSP for a fixed set of nodes.
    """

    def __init__(self, distance_matrix: np.ndarray, timeout_seconds: float = 10.0):
        self.distance_matrix = distance_matrix
        self.timeout_seconds = timeout_seconds

    def solve(self, assigned_nodes: List[int]) -> Tuple[bool, float, List[int]]:
        """
        Solves TSP for a given subset of nodes (including depot 0).

        Args:
            assigned_nodes: List of node indices to visit (0 must be included).

        Returns:
            Tuple of (is_feasible, min_distance, optimal_route)
        """
        if not OR_TOOLS_AVAILABLE:
            # Fallback or error
            return False, 0.0, []

        if not assigned_nodes:
            return True, 0.0, [0, 0]

        if 0 not in assigned_nodes:
            assigned_nodes = [0] + assigned_nodes

        # Mapping internal nodes to local indices for OR-Tools
        local_to_global = {i: node for i, node in enumerate(assigned_nodes)}
        num_local = len(assigned_nodes)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(num_local, 1, 0)

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            from_node = local_to_global[manager.IndexToNode(from_index)]
            to_node = local_to_global[manager.IndexToNode(to_index)]
            return int(self.distance_matrix[from_node][to_node] * 1000)  # Scaling for precision

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = int(self.timeout_seconds)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            dist = solution.ObjectiveValue() / 1000.0
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(local_to_global[manager.IndexToNode(index)])
                index = solution.Value(routing.NextVar(index))
            route.append(local_to_global[manager.IndexToNode(index)])
            return True, dist, route

        return False, 0.0, []
