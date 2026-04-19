"""
Cyclic Transfer (p-exchange) Operator Module.

Generalizes the swap operator across *p* routes.  Given p routes and one
selected node per route, the nodes are moved in a cyclic permutation:

    Route_0 donates node_0 → Route_1
    Route_1 donates node_1 → Route_2
    ...
    Route_{p-1} donates node_{p-1} → Route_0

The operator evaluates both forward and backward cyclic shifts and applies
the best improving configuration.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.inter_route.cyclic_transfer import cyclic_transfer
    >>> improved = cyclic_transfer(ls, [(r0, p0), (r1, p1), (r2, p2)])
"""

from typing import Any, List, Tuple

# (route_index, position_in_route)
RouteCut = Tuple[int, int]


def cyclic_transfer(ls: Any, participants: List[RouteCut]) -> bool:
    """
    Cyclic p-exchange: rotate nodes cyclically across p routes.

    Each participating route donates the node at the specified position and
    receives a node from another route in a cyclic pattern.  Both forward
    and backward cyclic shifts are evaluated.

    Args:
        ls: LocalSearch instance.
        participants: List of ``(route_index, node_position)`` tuples.
                      All route indices must be distinct.  Minimum 3.

    Returns:
        bool: True if an improving cyclic transfer was applied.

    Raises:
        ValueError: If fewer than 3 participants or duplicate routes.
    """
    p = len(participants)
    if p < 3:
        raise ValueError(f"Cyclic transfer requires >= 3 routes, got {p}")

    route_indices = [r for r, _ in participants]
    if len(set(route_indices)) != p:
        raise ValueError("All route indices in a cyclic transfer must be distinct")

    # Gather nodes and their demands
    nodes = []
    demands = []
    for r_idx, pos in participants:
        node = ls.routes[r_idx][pos]
        nodes.append(node)
        demands.append(ls.waste.get(node, 0))

    # Evaluate forward shift: route_i receives node from route_{i-1}
    forward_gain = _evaluate_shift(ls, participants, nodes, demands, direction=1)

    # Evaluate backward shift: route_i receives node from route_{i+1}
    backward_gain = _evaluate_shift(ls, participants, nodes, demands, direction=-1)

    best_gain = max(forward_gain, backward_gain)
    if best_gain * ls.C > 1e-4:
        direction = 1 if forward_gain >= backward_gain else -1
        _apply_shift(ls, participants, nodes, direction)
        return True

    return False


def _evaluate_shift(
    ls: Any,
    participants: List[RouteCut],
    nodes: List[int],
    demands: List[float],
    direction: int,
) -> float:
    """Compute the cost gain for a cyclic shift in the given direction."""
    p = len(participants)
    total_gain = 0.0

    for i in range(p):
        r_idx, pos = participants[i]
        route = ls.routes[r_idx]
        old_node = nodes[i]
        # In a forward shift (dir=1), route_i receives node from route_{i-1}
        donor_idx = (i - direction) % p
        new_node = nodes[donor_idx]

        # Check capacity
        load = ls._get_load_cached(r_idx)
        new_load = load - demands[i] + demands[donor_idx]
        if new_load > ls.Q:
            return -float("inf")

        # Delta: removal of old_node + insertion of new_node at same position
        prev_n = route[pos - 1] if pos > 0 else 0
        next_n = route[pos + 1] if pos < len(route) - 1 else 0

        removal = ls.d[prev_n, old_node] + ls.d[old_node, next_n]
        insertion = ls.d[prev_n, new_node] + ls.d[new_node, next_n]
        total_gain += removal - insertion

    return total_gain


def _apply_shift(
    ls: Any,
    participants: List[RouteCut],
    nodes: List[int],
    direction: int,
) -> None:
    """Apply the cyclic shift by replacing nodes in each route."""
    p = len(participants)
    affected = set()

    for i in range(p):
        r_idx, pos = participants[i]
        donor_idx = (i - direction) % p
        ls.routes[r_idx][pos] = nodes[donor_idx]
        affected.add(r_idx)

    ls._update_map(affected)
