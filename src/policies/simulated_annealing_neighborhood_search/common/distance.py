"""
Distance and time calculations for SANS.
"""

from logic.src.constants.routing import (
    COLLECTION_TIME_MINUTES,
    VEHICLE_SPEED_KMH,
)


def compute_distance_per_route(routes_list, distance_matrix):
    """
    Calculate the total travel distance for each individual route.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_matrix (np.ndarray): All-pairs shortest path distances.

    Returns:
        List[float]: Distance travelled by each vehicle/route.
    """
    travelled_distance_route = 0
    distance_route_vector = []
    for i in routes_list:
        for j, element in enumerate(i):
            if j < len(i) - 1:
                travelled_distance_route += distance_matrix[element][i[j + 1]]
        distance_route_vector.append(travelled_distance_route)
        travelled_distance_route = 0
    return distance_route_vector


def compute_route_time(routes_list, distance_route_vector):
    """
    Estimate total duration of each route (collection time + travel time).

    Assumes:
    - COLLECTION_TIME_MINUTES minutes per bin collection.
    - VEHICLE_SPEED_KMH km/h average vehicle speed.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_route_vector (List[float]): Distances per route.

    Returns:
        List[float]: Duration in minutes for each route.
    """
    route_time_vector = []
    for route_n, i in enumerate(routes_list):
        route_time = (len(i) - 2) * COLLECTION_TIME_MINUTES + (
            (distance_route_vector[route_n] / VEHICLE_SPEED_KMH) * 60
        )
        route_time_vector.append(route_time)
    return route_time_vector


def compute_sans_route_cost(route, distance_matrix, id_to_index):
    """
    Calculate the travel cost/distance for a single route using id-to-index mapping.

    Args:
        route (List[int]): Sequence of node IDs.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Mapping from node ID to matrix index.

    Returns:
        float: Route distance.
    """
    cost = 0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        cost += distance_matrix[id_to_index[a], id_to_index[b]]
    return cost


def compute_total_cost(routes, distance_matrix, id_to_index):
    """
    Calculate total travel distance for all routes.

    Args:
        routes (List[List[int]]): Set of routes.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Node ID to matrix index mapping.

    Returns:
        float: Total distance.
    """
    return sum(compute_sans_route_cost(route, distance_matrix, id_to_index) for route in routes)
