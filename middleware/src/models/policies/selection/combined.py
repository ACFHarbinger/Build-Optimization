"""
Combined Selection Strategy.
"""

import torch
from torch import Tensor

from .base import VectorizedSelector


class CombinedSelector(VectorizedSelector):
    """
    Combines multiple selection strategies with logical OR.

    A bin is selected if ANY of the constituent selectors select it.
    """

    def __init__(self, selectors: list[VectorizedSelector], logic: str = "or"):
        """
        Initialize CombinedSelector.

        Args:
            selectors: List of VectorizedSelector instances to combine.
            logic: logical operator to combine selectors ('or' or 'and'). Default: 'or'.
        """
        self.selectors = selectors
        self.logic = logic.lower()
        if self.logic not in ["or", "and"]:
            raise ValueError(f"Unknown logic: {self.logic}. Must be 'or' or 'and'.")

    def select(
        self,
        fill_levels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Select bins chosen by any constituent selector.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
            **kwargs: Passed to all constituent selectors.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        if self.logic == "or":
            combined = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
            for selector in self.selectors:
                mask = selector.select(fill_levels, **kwargs)
                combined = combined | mask
        else:  # logic == "and"
            combined = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
            for selector in self.selectors:
                mask = selector.select(fill_levels, **kwargs)
                combined = combined & mask

        return combined
