"""
Intra-route node relocation operators.

Provides procedures to move single or multiple bins to different positions
within the same route, supporting both random and consecutive relocations.
"""

from random import randint
from random import sample as rsample
from typing import Optional

__all__ = ["move_1_route", "move_n_route_random", "move_n_route_consecutive"]


def move_1_route(routes_list: list) -> None:
    """
    Move one bin to a different position in the same route.

    Args:
        routes_list: Current routing solution.
    """
    if not routes_list:
        return

    chosen_route = rsample(routes_list, 1)[0]
    if len(chosen_route) <= 3:
        return

    # Select bin to move
    bin_to_move = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

    # Select target bin to insert AT
    target_bin = bin_to_move
    while target_bin == bin_to_move:
        target_bin = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

    target_idx = chosen_route.index(target_bin)
    chosen_route.remove(bin_to_move)
    chosen_route.insert(target_idx, bin_to_move)


def move_n_route_random(routes_list: list, n: Optional[int] = None) -> Optional[int]:
    """
    Move n random bins to new positions within their routes.

    Args:
        routes_list: Current routing solution.
        n: Number of nodes to move.

    Returns:
        The number of nodes moved.
    """
    if not routes_list:
        return None

    chosen_route = rsample(routes_list, 1)[0]
    if n is None:
        n = rsample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) <= n + 2:
        return n

    for _ in range(n):
        # We need to re-sample every time because indices change
        # and we want to move 'n' nodes one after another as in the original code.
        bin_to_move = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

        target_bin = bin_to_move
        while target_bin == bin_to_move:
            target_bin = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

        target_idx = chosen_route.index(target_bin)
        chosen_route.remove(bin_to_move)
        chosen_route.insert(target_idx, bin_to_move)

    return n


def move_n_route_consecutive(routes_list: list, n: Optional[int] = None) -> Optional[int]:
    """
    Move a sequence of consecutive bins to a new position in the same route.

    Args:
        routes_list: Current routing solution.
        n: Number of consecutive nodes to move.

    Returns:
        The number of nodes moved.
    """
    if not routes_list:
        return None

    chosen_route = rsample(routes_list, 1)[0]
    if n is None:
        n = rsample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) <= n + 2:
        return n

    # Find a segment of size n (starting index should allow n nodes + depot)
    start_idx = randint(1, len(chosen_route) - n - 1)
    segment = [chosen_route.pop(start_idx) for _ in range(n)]

    # Target node to insert AT
    target_bin = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
    target_idx = chosen_route.index(target_bin)

    # Insert back at the new position
    for i, node in enumerate(segment):
        chosen_route.insert(target_idx + i, node)

    return n
