"""
Limited Split Algorithm (Max Vehicles).
"""

import torch

from .reconstruction import reconstruct_limited


def vectorized_split_limited(
    B,
    N,
    device,
    max_vehicles,
    capacity,
    cum_load,
    cum_dist_pad,
    d_0_i,
    d_i_0,
    giant_tours,
):
    """
    Limited split using K-step updates (like K shortest path layers).
    Constrains the solution to use at most `max_vehicles` routes.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        device: Torch device.
        max_vehicles (int): Maximum number of vehicles allowed.
        capacity (float): Vehicle capacity.
        cum_load (torch.Tensor): Cumulative load tensor.
        cum_dist_pad (torch.Tensor): Cumulative distance tensor (padded).
        d_0_i (torch.Tensor): Distance from depot to nodes.
        d_i_0 (torch.Tensor): Distance from nodes to depot.
        giant_tours (torch.Tensor): Giant tour indices.

    Returns:
        tuple: (routes_batch, costs)
    """
    indices = torch.arange(N + 1, device=device)
    J = indices.view(1, N + 1, 1).expand(B, N + 1, N + 1)
    idx_i = indices.view(1, 1, N + 1).expand(B, N + 1, N + 1)

    # Mask valid upper triangle: j < i
    mask_indices = idx_i > J

    # Load Constraint
    # L[b, j, i] = cum_load[b, i] - cum_load[b, j].
    L = cum_load.unsqueeze(1) - cum_load.unsqueeze(2)

    mask_cap = (capacity >= L) & mask_indices

    # Costs
    # d0_pad (B, N+1)
    d0_pad = torch.cat([d_0_i, torch.zeros((B, 1), device=device)], dim=1)  # (B, N+1)

    # Matrix Broadcast
    # D_Start[j, i] depends only on j.
    D_Start = d0_pad.unsqueeze(2).expand(-1, -1, N + 1)  # (B, N+1, N+1) along j

    # D_End[j, i] depends only on i.
    d_end_vec = torch.cat([torch.zeros((B, 1), device=device), d_i_0], dim=1)
    D_End = d_end_vec.unsqueeze(1).expand(-1, N + 1, -1)  # (B, N+1, N+1) along i

    # Path Dist[j, i] = cum_dist[i] - cum_dist[j+1]
    cd_pad = torch.cat([cum_dist_pad, torch.zeros((B, 1), device=device)], dim=1)  # (B, N+2)

    CD_I = cum_dist_pad.unsqueeze(1).expand(-1, N + 1, -1)

    cd_j_vec = cd_pad[:, 1 : N + 2]  # (B, N+1)
    CD_J = cd_j_vec.unsqueeze(2).expand(-1, -1, N + 1)

    PathCosts = CD_I - CD_J

    Costs = D_Start + PathCosts + D_End
    Costs[~mask_cap] = float("inf")

    # Optimization Loop
    V = torch.full((B, N + 1), float("inf"), device=device)
    V[:, 0] = 0

    # Storage for reconstruction
    P_k = torch.full((B, max_vehicles + 1, N + 1), -1, dtype=torch.long, device=device)

    best_cost = torch.full((B,), float("inf"), device=device)
    best_k_idx = torch.full((B,), -1, dtype=torch.long, device=device)

    for k in range(1, max_vehicles + 1):
        # V_new[i] = min_j (V_old[j] + Costs[j, i])
        M = V.unsqueeze(2) + Costs
        min_vals, min_idxs = M.min(dim=1)  # (B, i)

        V_next = min_vals

        # Store
        P_k[:, k, :] = min_idxs
        V = V_next

        # If V[:, N] is better than best_cost, update
        improved = V[:, N] < best_cost
        best_cost[improved] = V[improved, N]
        best_k_idx[improved] = k

    return reconstruct_limited(B, N, giant_tours, P_k, best_k_idx, best_cost)
