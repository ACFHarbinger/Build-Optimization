"""
Last Minute Selection Strategy.
"""

from typing import Optional

from torch import Tensor

from .base import VectorizedSelector


class LastMinuteSelector(VectorizedSelector):
    """
    Threshold-based reactive selection.

    Selects bins where current fill level exceeds the threshold.
    Simple but reactive - only collects when bins are nearly full.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize LastMinuteSelector.

        Args:
            threshold: Fill level threshold in [0, 1]. Default: 0.7 (70%).
        """
        self.threshold = threshold

    def select(
        self,
        fill_levels: Tensor,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins exceeding the fill threshold.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            threshold: Optional override for the fill threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        thresh = threshold if threshold is not None else self.threshold
        mandatory = fill_levels > thresh

        # Depot (index 0) is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
