"""
Intra-route swap operators for the Look-Ahead policy.

This module provides functions to perform intra-route swaps, such as swapping
two random bins or swapping sequences of bins within the same route.

Attributes:
    None

Example:
    >>> from logic.src.policies.simulated_annealing_neighborhood_search.operators.intra_swap import swap_1_route
    >>> swap_1_route(routes)
"""

from random import Random
from typing import List, Optional


def swap_1_route(routes_list: List[List[int]], rng: Random) -> None:
    """
    Intra-route perturbation: Swap two random bins within the same route.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        rng (Random): Random number generator.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
        if len(chosen_route) > 3:
            bins_to_swap = rng.sample(chosen_route[1 : len(chosen_route) - 1], 2)
            pos1 = chosen_route.index(bins_to_swap[0])
            pos2 = chosen_route.index(bins_to_swap[1])
            chosen_route[pos1], chosen_route[pos2] = chosen_route[pos2], chosen_route[pos1]


def swap_n_route_random(routes_list: List[List[int]], rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Intra-route perturbation: Swap n pairs of random bins within their routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        rng (Random): Random number generator.
        n (int, optional): Number of pairs to swap.

    Returns:
        int: The number of pairs swapped (chosen_n).
    """
    if len(routes_list) == 0:
        return None

    chosen_route = rng.sample(routes_list, 1)[0]
    chosen_n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) > chosen_n + 2:
        for _ in range(chosen_n):
            bins_to_swap = rng.sample(chosen_route[1 : len(chosen_route) - 1], 2)
            pos1 = chosen_route.index(bins_to_swap[0])
            pos2 = chosen_route.index(bins_to_swap[1])
            chosen_route[pos1], chosen_route[pos2] = chosen_route[pos2], chosen_route[pos1]

    return chosen_n


def swap_n_route_consecutive(routes_list: List[List[int]], rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Intra-route perturbation: Swap two consecutive sequences of n bins
    within the same route.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        rng (Random): Random number generator.
        n (int, optional): Number of consecutive nodes to swap.

    Returns:
        int: The number of consecutive nodes swapped (chosen_n).
    """
    if len(routes_list) == 0:
        return None

    chosen_route = rng.sample(routes_list, 1)[0]
    chosen_n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) > chosen_n + 2:

        def get_segments(route, n_size):
            """
            Identify two non-overlapping segments of length n_size in the route.

            Args:
                route (List[int]): The route to search.
                n_size (int): Length of segments to find.

            Returns:
                Tuple[int, int] | None: Start indices of the two segments, or None if not found.
            """
            # Pick a segment of size n_size (excluding start/end depots)
            # Route: [0, 1, 2, 3, 4, 0], len=6, bins=[1, 2, 3, 4]
            # range for start index: 1 to len-n_size-1
            max_start = len(route) - n_size - 1
            if max_start < 1:
                return None
            start1 = rng.sample(range(1, max_start + 1), 1)[0]
            indices1 = list(range(start1, start1 + n_size))

            # Find possible starts for second segment that don't overlap
            possible_starts2 = []
            for s2 in range(1, max_start + 1):
                indices2 = set(range(s2, s2 + n_size))
                if indices2.isdisjoint(set(indices1)):
                    possible_starts2.append(s2)

            if not possible_starts2:
                return None

            start2 = rng.sample(possible_starts2, 1)[0]
            return start1, start2

        segments = get_segments(chosen_route, chosen_n)
        if segments:
            s1, s2 = segments
            # Swap segments
            # For simplicity, we can do it element by element or slice
            seg1 = chosen_route[s1 : s1 + chosen_n]
            seg2 = chosen_route[s2 : s2 + chosen_n]
            chosen_route[s1 : s1 + chosen_n] = seg2
            chosen_route[s2 : s2 + chosen_n] = seg1

    return chosen_n
