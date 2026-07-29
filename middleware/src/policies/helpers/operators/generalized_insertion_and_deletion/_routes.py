from typing import List, Tuple


def _extract_working_route(route: List[int]) -> Tuple[int, bool, List[int], int]:
    """
    Extracts working route characteristics to handle open/closed route representations.

    Args:
        route: The tour as a list of node IDs.

    Returns:
        Tuple of (n, is_closed, work_route, n_work)
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)
    return n, is_closed, work_route, n_work
