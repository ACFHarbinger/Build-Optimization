"""
k-Permutation Operator Module (Open-loop TSP).

Extracts a sub-sequence of *k* consecutive nodes from a route and evaluates
all k! orderings, applying the one with the lowest cost.

Designed for open-loop routing where start/end depots are not necessarily
connected.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.intra_route.k_permutation import k_permutation
    >>> improved = k_permutation(ls, r_idx=0, start_pos=2, k=4)
"""

import itertools
from typing import Any, List, Optional, Tuple


def k_permutation(ls: Any, r_idx: int, start_pos: int, k: int = 3) -> bool:
    """
    k-permutation: reorder k consecutive nodes for minimum cost.

    Extracts the sub-sequence ``route[start_pos:start_pos+k]``, evaluates
    all ``k!`` orderings, and replaces the original with the best.

    Args:
        ls: LocalSearch instance.
        r_idx: Route index.
        start_pos: Starting position of the sub-sequence.
        k: Number of consecutive nodes to permute (default 3, max ~5).

    Returns:
        bool: True if a better ordering was found and applied.
    """
    route = ls.routes[r_idx]
    end_pos = start_pos + k

    if end_pos > len(route):
        return False
    if k < 2:
        return False

    sub = route[start_pos:end_pos]

    # Boundary nodes
    prev_node = route[start_pos - 1] if start_pos > 0 else 0
    next_node = route[end_pos] if end_pos < len(route) else 0

    # Original cost of connecting prev → sub → next
    original_cost = _subseq_cost(ls.d, prev_node, sub, next_node)

    best_cost = original_cost
    best_perm: Optional[Tuple[int, ...]] = None

    for perm in itertools.permutations(range(k)):
        if perm == tuple(range(k)):
            continue  # Skip identity
        reordered = [sub[i] for i in perm]
        cost = _subseq_cost(ls.d, prev_node, reordered, next_node)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is not None and (original_cost - best_cost) * ls.C > 1e-4:
        reordered = [sub[i] for i in best_perm]
        route[start_pos:end_pos] = reordered
        ls._update_map({r_idx})
        return True

    return False


def three_permutation(ls: Any, r_idx: int, start_pos: int) -> bool:
    """
    3-permutation: reorder 3 consecutive nodes for minimum cost.

    Convenience wrapper around :func:`k_permutation` with ``k=3``.

    Args:
        ls: LocalSearch instance.
        r_idx: Route index.
        start_pos: Starting position of the 3-node sub-sequence.

    Returns:
        bool: True if a better ordering was found and applied.
    """
    return k_permutation(ls, r_idx, start_pos, k=3)


def _subseq_cost(d, prev_node: int, seq: List[int], next_node: int) -> float:
    """Compute total edge cost of prev → seq[0] → ... → seq[-1] → next."""
    if not seq:
        return d[prev_node, next_node]
    cost = d[prev_node, seq[0]]
    for i in range(len(seq) - 1):
        cost += d[seq[i], seq[i + 1]]
    cost += d[seq[-1], next_node]
    return cost
