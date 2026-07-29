"""
Lookahead (Predictive) Selection Strategy.
"""

from typing import Optional

import torch
from torch import Tensor

from .base import VectorizedSelector


class LookaheadSelector(VectorizedSelector):
    """
    Predictive selection looking several days ahead.

    Selects bins that will overflow within the lookahead horizon based on
    current fill levels and accumulation rates.
    """

    def __init__(self, max_fill: float = 1.0):
        """
        Initialize LookaheadSelector.

        Args:
            max_fill: Maximum fill level (overflow threshold). Default: 1.0.
        """
        self.max_fill = max_fill

    def select(
        self,
        fill_levels: Tensor,
        accumulation_rates: Optional[Tensor] = None,
        max_fill: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins predicted to overflow within dynamic horizon.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            accumulation_rates: Daily fill rate (batch_size, num_nodes) in [0, 1].
            max_fill: Optional override for overflow threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        overflow_thresh = max_fill if max_fill is not None else self.max_fill
        if accumulation_rates is None:
            # Without rates, fall back to threshold-based selection
            mandatory = fill_levels >= overflow_thresh
        else:
            # 1. Identify bins overflowing today (initial selection)
            # current + rate >= max_fill
            initial_mandatory = (fill_levels + accumulation_rates) >= overflow_thresh

            # 2. Simulate collection: collected bins become 0.
            # Calculate days until next overflow for these bins: ceil(max_fill / rate)
            # Avoid division by zero
            rates_safe = accumulation_rates.clamp(min=1e-6)
            days_to_overflow = torch.ceil(overflow_thresh / rates_safe)

            # Mask: only consider days for initially selected bins
            # Non-selected bins effectively have infinite days (ignored)
            # We use a large number for infinity
            LARGE_NUM = 1e6
            next_overflow_days = torch.where(
                initial_mandatory, days_to_overflow, torch.tensor(LARGE_NUM, device=fill_levels.device)
            )

            # 3. Find the minimum next overflow day for each batch
            # Shape: (batch_size,)
            horizon = next_overflow_days.min(dim=1).values

            # 4. Check if other bins overflow before this horizon
            # Logic: check if (current + (horizon - 1) * rate) >= max_fill
            # But we must ensure horizon-1 >= 1 to be meaningful (look at least 1 day ahead)
            # Effectively, check_days = max(1.0, horizon - 1.0)
            check_days = (horizon - 1.0).clamp(min=1.0)

            # Broadcast check_days to (batch_size, 1) for multiplication
            check_days = check_days.unsqueeze(1)

            # Final check includes initial selection + new candidates
            predicted_fill = fill_levels + (check_days * accumulation_rates)
            mandatory = predicted_fill >= overflow_thresh

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False
        return mandatory
