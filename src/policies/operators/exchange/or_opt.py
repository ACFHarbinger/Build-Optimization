"""
Or-opt Operator Module.

This module implements the Or-opt operator for local search, which tries to
relocate chains of 1 to 3 consecutive nodes to better positions in the solution.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.exchange.or_opt import move_or_opt
    >>> improved = move_or_opt(ls, node=5, chain_len=2, r_idx=0, pos=1)
"""

from typing import Any


def move_or_opt(
    ls: Any,
    node: int,
    chain_len: int,
    r_idx: int,
    pos: int,
) -> bool:
    """
    Or-opt: Relocate a chain of consecutive nodes to the best position.

    Moves a sequence of 1-3 consecutive customers to another position,
    either within the same route or to a different route. Particularly
    effective when customers are geographically clustered.

    Args:
        ls: LocalSearch instance with routes, distance matrix, waste, etc.
        node: Starting node of the chain.
        chain_len: Length of chain to move (1, 2, or 3).
        r_idx: Route index containing the chain.
        pos: Position of the starting node in the route.

    Returns:
        bool: True if an improving move was found and applied.
    """
    route = ls.routes[r_idx]
    if pos + chain_len > len(route):
        return False

    # Extract chain
    chain = route[pos : pos + chain_len]
    chain_waste = sum(ls.waste.get(n, 0) for n in chain)

    # Calculate removal cost
    prev_node = route[pos - 1] if pos > 0 else 0
    next_node = route[pos + chain_len] if pos + chain_len < len(route) else 0
    removal_gain = ls.d[prev_node, chain[0]] + ls.d[chain[-1], next_node] - ls.d[prev_node, next_node]

    best_delta = 0.0
    best_insertion = None  # (route_idx, insert_pos)

    # Try all positions in all routes
    for target_r_idx, target_route in enumerate(ls.routes):
        # Check capacity
        if target_r_idx != r_idx:
            target_load = ls._calc_load_fresh(target_route)
            if target_load + chain_waste > ls.Q:
                continue

        # Try all insertion positions
        for insert_pos in range(len(target_route) + 1):
            # Skip positions that would just reinsert at same spot
            if target_r_idx == r_idx and insert_pos in range(pos, pos + chain_len + 1):
                continue

            ins_prev = target_route[insert_pos - 1] if insert_pos > 0 else 0
            ins_next = target_route[insert_pos] if insert_pos < len(target_route) else 0

            # Adjust for removal if same route
            if target_r_idx == r_idx and insert_pos > pos + chain_len:
                # Actually insert at insert_pos - chain_len after removal
                temp_route = route[:pos] + route[pos + chain_len :]
                adj_pos = insert_pos - chain_len
                ins_prev = temp_route[adj_pos - 1] if adj_pos > 0 else 0
                ins_next = temp_route[adj_pos] if adj_pos < len(temp_route) else 0

            insertion_cost = ls.d[ins_prev, chain[0]] + ls.d[chain[-1], ins_next] - ls.d[ins_prev, ins_next]

            delta = insertion_cost - removal_gain

            if delta < best_delta - 1e-6:
                best_delta = delta
                best_insertion = (target_r_idx, insert_pos)

    # Apply best move
    if best_insertion is not None:
        target_r, ins_pos = best_insertion

        # Remove chain from original route
        del ls.routes[r_idx][pos : pos + chain_len]

        # Adjust insertion position if same route and after removal point
        if target_r == r_idx and ins_pos > pos + chain_len:
            ins_pos -= chain_len

        # Insert chain
        for i, n in enumerate(chain):
            ls.routes[target_r].insert(ins_pos + i, n)

        ls._update_map({r_idx, target_r})
        return True

    return False
