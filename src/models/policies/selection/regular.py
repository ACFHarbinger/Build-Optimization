"""
Regular (Periodic) Selection Strategy.
"""

from typing import Optional

import torch
from torch import Tensor

from .base import VectorizedSelector


class RegularSelector(VectorizedSelector):
    """
    Periodic collection strategy.

    Selects all bins on scheduled collection days based on a fixed frequency.
    """

    def __init__(self, frequency: int = 3):
        """
        Initialize RegularSelector.

        Args:
            frequency: Collection interval in days. Default: 3 (collect every 3rd day).
        """
        self.frequency = frequency

    def select(
        self,
        fill_levels: Tensor,
        current_day: Optional[Tensor] = None,
        frequency: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select all bins if today is a scheduled collection day.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
            current_day: Current simulation day (batch_size,) or scalar.
            frequency: Optional override for collection frequency.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        freq = frequency if frequency is not None else self.frequency

        if freq <= 0:
            # Collect every day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        elif current_day is None:
            # No day info - assume collection day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        else:
            # Ensure current_day is a tensor
            if not isinstance(current_day, Tensor):
                current_day = torch.tensor(current_day, device=device)

            # Expand to batch if scalar
            if current_day.dim() == 0:
                current_day = current_day.expand(batch_size)

            # Collection day check: day % (freq + 1) == 1
            is_collection_day = (current_day % (freq + 1)) == 1  # (batch_size,)
            is_collection_day = is_collection_day.unsqueeze(-1)  # (batch_size, 1)

            # If collection day, all bins are mandatory
            mandatory = is_collection_day.expand(-1, num_nodes)

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
