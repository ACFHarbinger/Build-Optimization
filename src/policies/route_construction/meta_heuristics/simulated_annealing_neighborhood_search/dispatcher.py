"""
Simulated Annealing Neighborhood Search (SANS) Dispatcher Module.

This module contains the execution logic for Simulated Annealing Neighborhood Search (SANS).
It provides functions to run both the improved ('new') and original ('og') SA engines.

Reference:
    Jorge, D., Antunes, A. P., Ramos, T. R. P., & Barbosa-Povoa, A. P.
    "A hybrid metaheuristic for smart waste collection problems with
    workload concerns", 2022.
"""

import random
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic.src.data.processor import convert_to_dict
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search import (
    improved_simulated_annealing,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    create_points,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.solution_initialization import (
    compute_initial_solution,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.refinement.route_search import (
    find_solutions,
)
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import get_route_cost

from .params import SANSParams


def execute_new(
    policy: Any, params: SANSParams, **kwargs: Any
) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
    """Execute the improved simulated annealing engine.

    Args:
        policy: The SANSPolicy instance.
        params: Type-safe SANS parameters.
        **kwargs: Arguments passed to the policy's execute method.

    Returns:
        Tuple[List[int], float, float, Any, Any]: Tour, distance, profit, and metadata.
    """
    mandatory = kwargs.get("mandatory", [])
    early_result = policy._validate_mandatory(mandatory)
    if early_result is not None:
        return early_result

    bins = kwargs["bins"]
    distance_matrix = kwargs["distance_matrix"]
    coords = kwargs["coords"]
    area = kwargs.get("area", "Rio Maior")
    waste_type = kwargs.get("waste_type", "plastic")
    config = kwargs.get("config", {})

    # Load area parameters
    Q, R, C, values = policy._load_area_params(area, waste_type, config)
    policy._log_solver_params(values, kwargs)

    B = values.get("B", 0)
    V = values.get("V", 0)

    # Prepare data for SANS logic
    data = pd.DataFrame(
        {
            "ID": np.arange(1, bins.n + 1),
            "#bin": np.arange(1, bins.n + 1),
            "Stock": bins.c.astype("float32"),
            "Accum_Rate": bins.means.astype("float32"),
        }
    )
    # Add depot
    depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
    data = pd.concat([depot_row, data], ignore_index=True)

    coords_dict = convert_to_dict(coords)
    id_to_index = {i: i for i in range(len(distance_matrix))}

    initial_routes = compute_initial_solution(data, coords_dict, distance_matrix, Q, id_to_index)
    current_route = initial_routes[0] if initial_routes else [0, 0]

    # Ensure mandatory bins are in the route
    mandatory_1 = list(mandatory)
    route_set = set(current_route)
    for b in mandatory_1:
        if b not in route_set:
            current_route.insert(1, b)

    # Use type-safe Params
    rng = random.Random(kwargs.get("seed", params.seed))

    optimized_routes, best_profit, last_distance, _, _ = improved_simulated_annealing(
        [current_route],
        distance_matrix,
        params.time_limit,
        id_to_index,
        data,
        Q,
        params.T_init,
        params.T_min,
        params.alpha,
        params.iterations_per_T,
        R,
        V,
        B,
        C,
        mandatory_1,
        perc_bins_can_overflow=params.perc_bins_can_overflow,
        volume=V,
        density_val=B,
        max_vehicles=1,
        rng=rng,
    )

    tour = optimized_routes[0] if optimized_routes else [0, 0]
    return tour, last_distance, best_profit, kwargs.get("search_context"), kwargs.get("multi_day_context")


def execute_og(
    policy: Any, params: SANSParams, **kwargs: Any
) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
    """Execute the original LAC (look-ahead collection) engine.

    Args:
        policy: The SANSPolicy instance.
        params: Type-safe SANS parameters.
        **kwargs: Arguments passed to the policy's execute method.

    Returns:
        Tuple[List[int], float, Any]: Tour, distance, and metadata.
    """
    mandatory = kwargs.get("mandatory", [])
    early_result = policy._validate_mandatory(mandatory)
    if early_result is not None:
        return early_result

    bins = kwargs["bins"]
    distance_matrix = kwargs["distance_matrix"]
    coords = kwargs["coords"]
    new_data = kwargs.get("new_data")  # Expecting the dataframe
    area = kwargs.get("area", "Rio Maior")
    waste_type = kwargs.get("waste_type", "plastic")
    config = kwargs.get("config", {})

    # Load area parameters
    Q, R, C, area_values = policy._load_area_params(area, waste_type, config)
    policy._log_solver_params(area_values, kwargs)

    values = {
        "Q": Q,
        "R": R,
        "B": area_values.get("B", 0),
        "C": C,
        "V": params.V,
        "shift_duration": params.shift_duration,
        "perc_bins_can_overflow": params.perc_bins_can_overflow,
    }

    # mandatory bins are 0-based, find_solutions expects 1-based
    mandatory_1 = [b + 1 for b in mandatory]

    # Handle case where new_data might not be passed
    if new_data is None:
        # Build minimal dataframe similar to _execute_new
        new_data = pd.DataFrame(
            {
                "ID": np.arange(1, bins.n + 1),
                "#bin": np.arange(1, bins.n + 1),
                "Stock": bins.c.astype("float32"),
                "Accum_Rate": bins.means.astype("float32"),
            }
        )
        depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
        new_data = pd.concat([depot_row, new_data], ignore_index=True)

    points = create_points(new_data, coords)

    rng = random.Random(kwargs.get("seed", params.seed))
    np_rng = np.random.default_rng(kwargs.get("seed", params.seed))
    try:
        res, _, _ = find_solutions(
            new_data,
            coords,
            distance_matrix,
            params.combination,
            mandatory_1,
            values,
            bins.n,
            points,
            time_limit=params.time_limit,
            rng=rng,
            np_rng=np_rng,
        )
    except Exception:
        return [0, 0], 0.0, 0.0, kwargs.get("search_context"), kwargs.get("multi_day_context")

    tour = res[0] if res else [0, 0]
    cost = get_route_cost(distance_matrix, tour)

    # Compute profit: collected revenue - distance cost
    visited = {n for n in tour if n != 0}
    collected_revenue = sum(float(bins.c[n - 1]) * R for n in visited if 1 <= n <= bins.n)
    profit = collected_revenue - cost * C

    return tour, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")
