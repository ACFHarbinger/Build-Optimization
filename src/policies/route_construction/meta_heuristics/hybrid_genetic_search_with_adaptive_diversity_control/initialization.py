"""
Initialization module for HGS-ADC Generation 0 via Multi-Period Clarke-Wright Savings.
"""

import numpy as np

from .individual import Individual
from .split import compute_daily_loads


def generate_initial_individual(  # noqa: C901
    N: int, T: int, base_wastes: np.ndarray, daily_increments: np.ndarray, dist: np.ndarray, capacity: float
) -> Individual:
    """
    Creates an Individual using Multi-Period Clarke-Wright savings.

    1. Assign random unconstrained patterns.
    2. Compute loads.
    3. Construct daily routes using C-W savings.
    4. Form giant tours by concatenating the routes.
    """
    # 1. Random pattern assignment
    # Search space is [0, 2^T - 1]. Depot (0) does not matter, give it 0 or full.
    patterns = np.random.randint(0, 1 << T, size=N)
    patterns[0] = 0  # Depot does not have a pattern conventionally

    # 2. Accumulated demand
    loads = compute_daily_loads(patterns, base_wastes, daily_increments, T)

    giant_tours = []

    # 3. Day-Specific C-W Savings
    for t in range(T):
        # Identify active nodes
        active_nodes = []
        for i in range(1, N):
            if (patterns[i] >> t) & 1:
                active_nodes.append(i)

        if not active_nodes:
            giant_tours.append(np.array([], dtype=int))
            continue

        # Initial routes: 0 -> i -> 0
        routes = [[i] for i in active_nodes]
        route_loads = {i: loads[t, i] for i in active_nodes}
        node_to_route = {i: i for i in active_nodes}  # Maps node ID to its route representative (start of list)

        # Calculate savings S_ij = d_0i + d_j0 - d_ij
        savings = []
        for i in active_nodes:
            for j in active_nodes:
                if i != j:
                    s = dist[0, i] + dist[j, 0] - dist[i, j]
                    savings.append((s, i, j))

        # Sort savings descending
        savings.sort(key=lambda x: x[0], reverse=True)

        # Build routes
        for _s, i, j in savings:
            route_i = node_to_route[i]
            route_j = node_to_route[j]

            # Must be different routes
            if route_i == route_j:
                continue

            # i must be end of route_i, j must be start of route_j
            # We track actual lists. Finding if i is end:
            list_i = None
            list_j = None

            for r in routes:
                if r and r[-1] == i:
                    list_i = r
                if r and r[0] == j:
                    list_j = r

            if list_i is not None and list_j is not None and list_i != list_j:
                # Capacity check
                load_i = sum(route_loads[node] for node in list_i)
                load_j = sum(route_loads[node] for node in list_j)

                # In HGS we could allow constraint violations conceptually,
                # but standard initialization respects capacity to start with feasible solutions if possible.
                if load_i + load_j <= capacity:
                    # Merge list_j into list_i
                    list_i.extend(list_j)
                    # Remove list_j from routes
                    routes.remove(list_j)
                    # Update node representations
                    for node in list_j:
                        node_to_route[node] = route_i

        # 4. Form giant tour by concatenation
        day_t_giant_tour = []
        for r in routes:
            day_t_giant_tour.extend(r)

        giant_tours.append(np.array(day_t_giant_tour, dtype=int))

    return Individual(patterns=patterns, giant_tours=giant_tours)
