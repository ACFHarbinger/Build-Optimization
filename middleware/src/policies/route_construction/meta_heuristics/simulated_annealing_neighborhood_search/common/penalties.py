"""
Cost and penalty calculations for SANS.
"""

from .distance import compute_route_time


def compute_transportation_cost(routes_list, distance_route_vector, C):
    """
    Calculate the total transportation cost based on distance and cost coefficient.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_route_vector (List[float]): Distances per route.
        C (float): Cost per distance unit.

    Returns:
        float: Total transportation cost.
    """
    total_travelled_distance = sum(distance_route_vector)
    transportation_cost = C * total_travelled_distance
    return transportation_cost


def compute_vehicle_use_penalty(routes_list, p_vehicle):
    """
    Calculate the penalty for the number of vehicles (routes) used.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Penalty coefficient per vehicle.

    Returns:
        float: Total vehicle penalty.
    """
    number_vehicles = len(routes_list)
    vehicle_penalty = p_vehicle * number_vehicles
    return vehicle_penalty


def compute_route_time_difference_penalty(routes_list, p_route_difference, distance_route_vector):
    """
    Calculate penalty for workload imbalance between routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_route_difference (float): Penalty coefficient for time difference.
        distance_route_vector (List[float]): Distances per route.

    Returns:
        float: Calculated imbalance penalty.
    """
    route_time_vector = compute_route_time(routes_list, distance_route_vector)
    if len(route_time_vector) >= 1:
        min_route_time = min(route_time_vector)
        max_route_time = max(route_time_vector)
        length_difference = max_route_time - min_route_time
        route_difference_penalty = p_route_difference * length_difference
    else:
        route_difference_penalty = 0

    return route_difference_penalty


def compute_shift_excess_penalty(routes_list, p_shift, distance_matrix, distance_route_vector, shift_duration):
    """
    Calculate penalty for routes exceeding the shift duration.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_shift (float): Penalty coefficient for overtime.
        distance_matrix (np.ndarray): Distance matrix.
        distance_route_vector (List[float]): Distances per route.
        shift_duration (float): Maximum allowed shift time (minutes).

    Returns:
        float: Total shift balance penalty.
    """
    route_time_vector = compute_route_time(routes_list, distance_route_vector)
    shift_excess = 0
    shift_excess_vector = []
    for i in route_time_vector:
        shift_excess = i - shift_duration
        if shift_excess <= 0:
            shift_excess_vector.append(0)
        else:
            shift_excess_vector.append(shift_excess)

        shift_excess = 0
    total_shift_excess = sum(shift_excess_vector)
    shift_excess_penalty = p_shift * total_shift_excess
    return shift_excess_penalty


def compute_load_excess_penalty(routes_list, p_load, data, vehicle_capacity):
    """
    Calculate penalty for vehicles exceeding their capacity limit.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_load (float): Penalty coefficient for excess load.
        data (pd.DataFrame): Bin weights data.
        vehicle_capacity (float): Max vehicle capacity.

    Returns:
        float: Total load penalty.
    """
    load_route = 0
    load_excess_route = 0
    total_load_excess = 0
    for i in routes_list:
        for j in i:
            stock = data["Stock"][j]
            accumulation_rate = data["Accum_Rate"][j]
            bin_stock = stock + accumulation_rate
            load_route += bin_stock
        load_excess_route = load_route - vehicle_capacity
        if load_excess_route > 0:
            total_load_excess += load_excess_route
        load_route = 0
        load_excess_route = 0
    load_excess_penalty = p_load * total_load_excess
    return load_excess_penalty
