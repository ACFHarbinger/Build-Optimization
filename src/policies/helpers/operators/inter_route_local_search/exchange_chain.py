"""
Exchange Chain Inter-Route Operator Module.

Provides inter-route operators for moving and swapping consecutive-node chains:

- ``exchange_2_0``: Move 2 consecutive nodes from one route to another.
- ``exchange_2_1``: Swap 2 consecutive nodes with 1 node across routes.
- ``exchange_k_0``: Generalised — move *k* consecutive nodes inter-route.
- ``exchange_k_h``: Generalised — swap *k* consecutive nodes with *h*
  consecutive nodes across routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.inter_route.exchange_chain import exchange_k_0
    >>> improved = exchange_k_0(ls, r_src=0, pos_src=1, r_dst=1, pos_dst=2, k=3)
"""

from typing import Any, List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_edge_cost(d, prev_node: int, chain: List[int], next_node: int) -> float:
    """Cost of edges: prev → chain[0] → … → chain[-1] → next."""
    if not chain:
        return d[prev_node, next_node]
    cost = d[prev_node, chain[0]]
    for i in range(len(chain) - 1):
        cost += d[chain[i], chain[i + 1]]
    cost += d[chain[-1], next_node]
    return cost


# ---------------------------------------------------------------------------
# Fixed-size operators (fast-path specialisations)
# ---------------------------------------------------------------------------


def exchange_2_0(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Exchange (2,0): move 2 consecutive nodes to another route.

    Removes nodes at positions ``pos_src`` and ``pos_src + 1`` from
    route ``r_src`` and inserts them after position ``pos_dst`` in route
    ``r_dst``.

    Args:
        ls: LocalSearch instance.
        r_src: Source route index.
        pos_src: Position of the first node in the chain.
        r_dst: Destination route index.
        pos_dst: Insertion position in destination (chain inserted after this).

    Returns:
        bool: True if the relocation improved the solution.
    """
    return exchange_k_0(ls, r_src, pos_src, r_dst, pos_dst, k=2)


def exchange_2_1(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Exchange (2,1): swap 2 consecutive nodes with 1 node across routes.

    Removes a chain of 2 consecutive nodes from ``r_src`` and a single
    node from ``r_dst``, then inserts each chain/node into the other
    route's gap.

    Args:
        ls: LocalSearch instance.
        r_src: Route with the 2-node chain.
        pos_src: Position of the first node in the chain.
        r_dst: Route with the single node.
        pos_dst: Position of the single node.

    Returns:
        bool: True if the swap improved the solution.
    """
    return exchange_k_h(ls, r_src, pos_src, 2, r_dst, pos_dst, 1)


# ---------------------------------------------------------------------------
# Generalised operators
# ---------------------------------------------------------------------------


def exchange_k_0(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int, k: int = 2) -> bool:
    """
    Exchange (k,0): move *k* consecutive nodes to another route.

    Removes the chain ``route_src[pos_src : pos_src + k]`` and inserts it
    after position ``pos_dst`` in route ``r_dst``.

    Args:
        ls: LocalSearch instance.
        r_src: Source route index.
        pos_src: Position of the first node in the chain.
        r_dst: Destination route index.
        pos_dst: Insertion position in destination (chain inserted after).
        k: Chain length (>= 1).

    Returns:
        bool: True if the relocation improved the solution.
    """
    if r_src == r_dst or k < 1:
        return False

    route_src = ls.routes[r_src]
    route_dst = ls.routes[r_dst]

    if pos_src + k > len(route_src):
        return False

    chain = route_src[pos_src : pos_src + k]

    # Capacity check
    dem_chain = sum(ls.waste.get(n, 0) for n in chain)
    if ls._get_load_cached(r_dst) + dem_chain > ls.Q:
        return False

    # Removal delta
    prev_c = route_src[pos_src - 1] if pos_src > 0 else 0
    next_c = route_src[pos_src + k] if pos_src + k < len(route_src) else 0

    removal = _chain_edge_cost(ls.d, prev_c, chain, next_c)
    repair = ls.d[prev_c, next_c]

    # Insertion delta
    v = route_dst[pos_dst]
    v_next = route_dst[pos_dst + 1] if pos_dst + 1 < len(route_dst) else 0

    old_edge = ls.d[v, v_next]
    insertion = _chain_edge_cost(ls.d, v, chain, v_next)

    delta = (repair - removal) + (insertion - old_edge)

    if delta * ls.C < -1e-4:
        # Remove chain from source (pop from end to preserve indices)
        for i in range(k - 1, -1, -1):
            route_src.pop(pos_src + i)
        # Insert chain into destination
        for i, node in enumerate(chain):
            route_dst.insert(pos_dst + 1 + i, node)
        ls._update_map({r_src, r_dst})
        return True

    return False


def exchange_k_h(
    ls: Any,
    r_src: int,
    pos_src: int,
    k: int,
    r_dst: int,
    pos_dst: int,
    h: int,
) -> bool:
    """
    Exchange (k,h): swap *k* consecutive nodes with *h* consecutive nodes.

    Removes a chain of *k* consecutive nodes from ``r_src`` and a chain of
    *h* consecutive nodes from ``r_dst``, then inserts each chain into the
    other route's gap.

    Args:
        ls: LocalSearch instance.
        r_src: Route with the k-node chain.
        pos_src: Position of the first node in the k-chain.
        k: Length of the source chain (>= 1).
        r_dst: Route with the h-node chain.
        pos_dst: Position of the first node in the h-chain.
        h: Length of the destination chain (>= 1).

    Returns:
        bool: True if the swap improved the solution.
    """
    if r_src == r_dst or k < 1 or h < 1:
        return False

    route_src = ls.routes[r_src]
    route_dst = ls.routes[r_dst]

    if pos_src + k > len(route_src):
        return False
    if pos_dst + h > len(route_dst):
        return False

    chain_k = route_src[pos_src : pos_src + k]
    chain_h = route_dst[pos_dst : pos_dst + h]

    dem_k = sum(ls.waste.get(n, 0) for n in chain_k)
    dem_h = sum(ls.waste.get(n, 0) for n in chain_h)

    # Capacity check after swap
    new_load_src = ls._get_load_cached(r_src) - dem_k + dem_h
    new_load_dst = ls._get_load_cached(r_dst) - dem_h + dem_k

    if new_load_src > ls.Q or new_load_dst > ls.Q:
        return False

    # Source: remove chain_k, insert chain_h
    prev_k = route_src[pos_src - 1] if pos_src > 0 else 0
    next_k = route_src[pos_src + k] if pos_src + k < len(route_src) else 0

    cost_remove_src = _chain_edge_cost(ls.d, prev_k, chain_k, next_k)
    cost_insert_src = _chain_edge_cost(ls.d, prev_k, chain_h, next_k)

    # Destination: remove chain_h, insert chain_k
    prev_h = route_dst[pos_dst - 1] if pos_dst > 0 else 0
    next_h = route_dst[pos_dst + h] if pos_dst + h < len(route_dst) else 0

    cost_remove_dst = _chain_edge_cost(ls.d, prev_h, chain_h, next_h)
    cost_insert_dst = _chain_edge_cost(ls.d, prev_h, chain_k, next_h)

    delta = (cost_insert_src - cost_remove_src) + (cost_insert_dst - cost_remove_dst)

    if delta * ls.C < -1e-4:
        # Replace source chain with destination chain
        route_src[pos_src : pos_src + k] = chain_h
        # Replace destination chain with source chain
        route_dst[pos_dst : pos_dst + h] = chain_k
        ls._update_map({r_src, r_dst})
        return True

    return False
