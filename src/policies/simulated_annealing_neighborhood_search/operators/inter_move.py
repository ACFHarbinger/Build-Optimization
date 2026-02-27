"""
Inter-route node relocation operators.

Provides procedures to move single or multiple bins between different routes,
supporting both random and consecutive relocations.
"""

from random import randint
from random import sample as rsample
from typing import Optional

__all__ = ["move_2_routes", "move_n_2_routes_random", "move_n_2_routes_consecutive"]


def move_2_routes(routes_list: list) -> None:
    """
    Inter-route perturbation: Move one bin from one route to another.

    Args:
        routes_list: Current routing solution.
    """
    if len(routes_list) < 2:
        return

    # Select two distinct routes
    r1, r2 = rsample(routes_list, 2)
    donor_route, receptor_route = rsample([r1, r2], 2)

    if len(donor_route) <= 2:
        return

    # Select bin to move from donor
    donor_bin = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

    # Select position in receptor (if receptor is empty/just-depots, use direct insert)
    if len(receptor_route) > 2:
        receptor_position = receptor_route.index(rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0])
    else:
        receptor_position = 1

    donor_route.remove(donor_bin)
    receptor_route.insert(receptor_position, donor_bin)

    # Clean up empty routes (length 2 means only depots)
    if len(donor_route) == 2:
        routes_list.remove(donor_route)


def move_n_2_routes_random(routes_list: list) -> Optional[int]:
    """
    Inter-route perturbation: Move n random bins from one route to another.

    Args:
        routes_list: Current routing solution.

    Returns:
        The number of nodes moved.
    """
    if len(routes_list) < 2:
        return None

    # Pick n
    n = rsample([2, 3, 4, 5], 1)[0]

    # Select donor and receptor
    r1, r2 = rsample(routes_list, 2)
    donor_route, receptor_route = rsample([r1, r2], 2)

    if len(donor_route) < n + 2:
        return n

    for _ in range(n):
        # Sample donor bin
        donor_bin = rsample(donor_route[1 : len(donor_route) - 1], 1)[0]

        # Sample receptor position
        if len(receptor_route) > 2:
            receptor_position = receptor_route.index(rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0])
        else:
            receptor_position = 1

        donor_route.remove(donor_bin)
        receptor_route.insert(receptor_position, donor_bin)

    # Clean up
    if len(donor_route) == 2:
        routes_list.remove(donor_route)

    return n


def move_n_2_routes_consecutive(routes_list: list) -> Optional[int]:
    """
    Inter-route perturbation: Move a sequence of consecutive bins from one route
    to another.

    Args:
        routes_list: Current routing solution.

    Returns:
        The number of nodes moved.
    """
    if len(routes_list) < 2:
        return None

    n = rsample([2, 3, 4, 5], 1)[0]

    r1, r2 = rsample(routes_list, 2)
    donor_route, receptor_route = rsample([r1, r2], 2)

    if len(donor_route) < n + 2:
        return n

    # Find segment in donor
    start_idx = randint(1, len(donor_route) - n - 1)
    segment = [donor_route.pop(start_idx) for _ in range(n)]

    # Target position in receptor
    if len(receptor_route) > 2:
        target_node = rsample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
        target_idx = receptor_route.index(target_node)
    else:
        target_idx = 1

    # Insert back
    for i, node in enumerate(segment):
        receptor_route.insert(target_idx + i, node)

    # Clean up
    if len(donor_route) == 2:
        routes_list.remove(donor_route)

    return n
