"""
Objective functions for SANS.
"""

from logic.src.constants.routing import MAX_CAPACITY_PERCENT

from .distance import compute_distance_per_route, compute_sans_route_cost
from .penalties import (
    compute_load_excess_penalty,
    compute_route_time_difference_penalty,
    compute_shift_excess_penalty,
    compute_transportation_cost,
    compute_vehicle_use_penalty,
)
from .revenue import compute_waste_collection_revenue


def compute_profit(
    routes_list,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Calculate the total objective value (Profit - Penalties).

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty coefficient.
        p_load (float): Load penalty coefficient.
        p_route_difference (float): Imbalance penalty coefficient.
        p_shift (float): Shift penalty coefficient.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary containing keys like E, B, R, C,
            vehicle_capacity, and shift_duration.

    Returns:
        float: Total calculated profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)
    route_difference_penalty = compute_route_time_difference_penalty(
        routes_list, p_route_difference, distance_route_vector
    )
    load_excess_penalty = compute_load_excess_penalty(routes_list, p_load, data, values["vehicle_capacity"])
    shift_excess_penalty = compute_shift_excess_penalty(
        routes_list,
        p_shift,
        distance_matrix,
        distance_route_vector,
        values["shift_duration"],
    )

    profit = (
        total_revenue
        - transportation_cost
        - vehicle_penalty
        - route_difference_penalty
        - load_excess_penalty
        - shift_excess_penalty
    )
    return profit


def compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values):
    """
    Calculate the "real" profit (Revenue - Travel Cost - Vehicle penalty)
    excluding soft constraints.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary.

    Returns:
        float: Physical profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)

    profit = total_revenue - transportation_cost - vehicle_penalty
    return profit


def compute_total_profit(routes, distance_matrix, id_to_index, data, R, V, density, cost_per_km=1.0):
    """
    Calcula o lucro total considerando:
    - Receita: soma dos pesos (kg) coletados multiplicado por R (€/kg)
    - Custo: soma das distâncias multiplicada por custo por km (default = 1.0 €/km)
    """
    total_kg = 0
    total_km = 0
    stocks = dict(zip(data["#bin"], data["Stock"]))
    for route in routes:
        if len(route) <= 2:
            continue
        # Calculate profit
        total_kg += (sum(stocks.get(b, 0) for b in route if b != 0) / MAX_CAPACITY_PERCENT) * V * (density)
        # Use compute_sans_route_cost for distance calculation to reuse logic
        total_km += compute_sans_route_cost(route, distance_matrix, id_to_index)

    receita = total_kg * R
    custo = total_km * cost_per_km
    lucro = receita - custo
    return lucro, total_kg, total_km, receita
