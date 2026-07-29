"""
Route Reconstruction Algorithms for Split.
"""


def reconstruct_routes(B, N, giant_tours, P, costs):
    """
    Reconstructs the routes from the predecessor matrix P obtained from the Split algorithm.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        giant_tours (torch.Tensor): Giant tour indices.
        P (torch.Tensor): Predecessor matrix (B, N+1).
        costs (torch.Tensor): Costs vector (B,).

    Returns:
        tuple: (list of routes per batch item, costs)
    """
    routes_batch = []
    P_cpu = P.cpu().numpy()
    giant_tours_cpu = giant_tours.cpu().numpy()

    for b in range(B):
        curr = N
        route_nodes = []  # type: ignore[var-annotated]
        possible = True
        while curr > 0:
            prev = P_cpu[b, curr]
            if prev == -1:
                possible = False
                break
            nodes = giant_tours_cpu[b, prev:curr]
            route_nodes = [0] + list(nodes) + route_nodes
            curr = prev

        if route_nodes and route_nodes[-1] != 0:
            route_nodes.append(0)

        if not possible:
            # Fallback or empty?
            pass

        routes_batch.append(route_nodes)
    return routes_batch, costs


def reconstruct_limited(B, N, giant_tours, P_k, best_k, costs):
    """
    Reconstructs routes for the limited vehicle split algorithm.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        giant_tours (torch.Tensor): Giant tour indices.
        P_k (torch.Tensor): Predecessor matrix with k dimension (B, max_k + 1, N + 1).
        best_k (torch.Tensor): Index of the best number of vehicles used for each batch item.
        costs (torch.Tensor): Costs vector.

    Returns:
        tuple: (list of routes, costs)
    """
    routes_batch = []  # type: ignore[var-annotated]
    P_cpu = P_k.cpu().numpy()
    k_cpu = best_k.cpu().numpy()
    giant_tours_cpu = giant_tours.cpu().numpy()

    for b in range(B):
        k = k_cpu[b]
        if k == -1 or costs[b] == float("inf"):
            routes_batch.append([])
            continue

        curr = N
        route_nodes = []  # type: ignore[var-annotated]

        # Backtrack with known k
        current_k = k

        while curr > 0 and current_k > 0:
            prev = P_cpu[b, current_k, curr]
            if prev == -1:
                break

            nodes = giant_tours_cpu[b, prev:curr]
            route_nodes = [0] + list(nodes) + route_nodes

            curr = prev
            current_k -= 1

        if route_nodes and route_nodes[-1] != 0:
            route_nodes.append(0)

        routes_batch.append(route_nodes)

    return routes_batch, costs
