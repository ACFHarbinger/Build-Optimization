"""
Evaluation metrics for HGS.
"""

import torch


def calc_broken_pairs_distance(population: torch.Tensor) -> torch.Tensor:
    """
    Computes average Broken Pairs Distance for each individual in the population.
    Distance(A, B) = 1 - (|Edges(A) inter Edges(B)| / N)

    Args:
        population: (B, P, N) tensor of giant tours.

    Returns:
        diversity: (B, P) diversity score (higher is better/more distant).
    """
    B, P, N = population.size()
    device = population.device

    # 1. Construct Edge Hashes
    # Edges: (i, i+1) and (N-1, 0)
    # Hash: min(u,v)*N + max(u,v) (assuming max node index < N? No, nodes are 1..N usually, or indices 0..N-1?)
    # giant_tours usually contains indices.

    # Create cyclic view
    next_nodes = torch.roll(population, shifts=-1, dims=2)

    # Sort u,v to be direction agnostic
    u = torch.min(population, next_nodes)
    v = torch.max(population, next_nodes)

    # Hash (N_max is safely larger than N, e.g. N+1)
    # Be careful if nodes are indices (0..N-1) or IDs (1..N).
    # If 0..N-1, then max hash approx N^2.
    hashes = u * (N + 100) + v  # (B, P, N)

    # 2. Compute Pairwise Distances
    # We want diversity[b, i] = mean_{j != i} (1 - intersection(i, j) / N)
    # Doing full PxP on GPU might be heavy if P is large (e.g. 100).
    # But for P=10-50, B=128, N=100:
    # (B, P, 1, N) == (B, 1, P, N) -> (B, P, P, N) comparison
    # Memory: 128 * 50 * 50 * 100 * 1 byte (bool) = 32 MB. Very Safe.

    # Expand for broadcast
    hashes.unsqueeze(2).unsqueeze(4)  # (B, P, 1, N, 1)
    hashes.unsqueeze(1).unsqueeze(3)  # (B, 1, P, 1, N)

    # Match matrix: matches[b, i, j, k, l]
    # intersections = torch.zeros((B, P, P), device=device)
    # Optimized:
    # hashes: (B, P, N)
    # (B, P, 1, N) == (B, 1, P, N) -> (B, P, P, N)

    # We can do this per batch if memory is tight, but P=50 is small.
    intersections = torch.zeros((B, P, P), device=device)
    for i in range(P):
        target = hashes[:, i : i + 1, :]
        matches = hashes.unsqueeze(3) == target.unsqueeze(2)
        num_shared = matches.any(dim=3).sum(dim=2)  # (B, P)
        intersections[:, i, :] = num_shared

    # Distance = 1 - (intersection / N)
    dists = 1.0 - (intersections.float() / N)

    # Diversity of i = mean distance to others
    diversity = dists.sum(dim=2) / max(1, P - 1)
    return diversity
