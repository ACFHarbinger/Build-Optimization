"""
Genetic Crossover Operators for HGS.
"""

from typing import Any, Optional

import torch


def vectorized_ordered_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    device: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized Ordered Crossover (OX1) with shared cuts across batch.

    Args:
        parent1: (B, N)
        parent2: (B, N)
        device (Optional[Any]): Torch device. If None, uses parent1.device.
        generator (Optional[torch.Generator]): Torch random number generator.

    Returns:
        offspring: (B, N)
    """
    B, N = parent1.size()
    device = parent1.device

    # 1. Generate shared cut points
    idx1, idx2 = torch.randint(0, N, (2,), device=device, generator=generator).tolist()
    start = min(idx1, idx2)
    end = max(idx1, idx2)

    if start == end:
        end = min(start + 1, N)

    num_seg = end - start
    num_rem = N - num_seg

    # 2. Extract segment from Parent 1
    # segment: (B, num_seg)
    segment = parent1[:, start:end]

    # 3. Create Offspring container
    offspring = torch.zeros_like(parent1)
    # Copy segment
    offspring[:, start:end] = segment

    # 4. Fill remaining from Parent 2
    # We need to take elements from P2 that are NOT in segment, maintaining order
    # starting from 'end' index (wrapping around).

    # Roll P2 so it starts at 'end'
    # logical shift: indices [end, end+1, ..., N-1, 0, ..., end-1]
    roll_idx = torch.arange(N, device=device)
    roll_idx = (roll_idx + end) % N
    p2_rolled = parent2[:, roll_idx]  # (B, N) sorted by fill order

    # Efficient exclusion check:
    # (B, N, 1) == (B, 1, num_seg) -> (B, N, num_seg) -> sum/any -> (B, N) mask
    # This uses B*N*num_seg memory. For N=100, B=128, this is ~1.2M elements (bool). Cheap.

    exists_in_seg = (p2_rolled.unsqueeze(2) == segment.unsqueeze(1)).any(dim=2)  # (B, N)

    # We want elements where ~exists_in_seg
    # Handle each batch element separately to be robust to duplicates
    fill_idx = torch.cat(
        [
            torch.arange(end, N, device=device),
            torch.arange(0, start, device=device),
        ]
    )

    for b in range(B):
        valid_mask = ~exists_in_seg[b]
        valid_vals_b = p2_rolled[b][valid_mask]

        # Pad or truncate to match num_rem
        if len(valid_vals_b) < num_rem:
            # Pad with values from parent1 that aren't in segment
            missing = num_rem - len(valid_vals_b)
            # Use first missing values from parent1 outside segment
            extra = parent1[b][
                torch.cat(
                    [
                        torch.arange(0, start, device=device),
                        torch.arange(end, N, device=device),
                    ]
                )
            ][:missing]
            valid_vals_b = torch.cat([valid_vals_b, extra])
        elif len(valid_vals_b) > num_rem:
            valid_vals_b = valid_vals_b[:num_rem]

        offspring[b, fill_idx] = valid_vals_b

    return offspring
