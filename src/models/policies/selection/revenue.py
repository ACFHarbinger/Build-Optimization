"""
Revenue Selection Strategy.
"""

from typing import Optional

from torch import Tensor

from .base import VectorizedSelector


class RevenueSelector(VectorizedSelector):
    """
    Revenue-based selection strategy.

    Selects bins where expected collection revenue exceeds a threshold.
    """

    def __init__(
        self,
        revenue_kg: float = 1.0,
        bin_capacity: float = 1.0,
        threshold: float = 0.0,
    ):
        """
        Initialize RevenueSelector.

        Args:
            revenue_kg: Revenue per kg of collected waste.
            bin_capacity: Capacity of each bin in kg.
            threshold: Minimum revenue threshold for selection.
        """
        self.revenue_kg = revenue_kg
        self.bin_capacity = bin_capacity
        self.threshold = threshold

    def select(
        self,
        fill_levels: Tensor,
        revenue_kg: Optional[float] = None,
        bin_capacity: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins where expected revenue exceeds threshold.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            revenue_kg: Optional override for revenue per kg.
            bin_capacity: Optional override for bin capacity.
            threshold: Optional override for revenue threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        rev = revenue_kg if revenue_kg is not None else self.revenue_kg
        cap = bin_capacity if bin_capacity is not None else self.bin_capacity
        thresh = threshold if threshold is not None else self.threshold

        # Expected revenue = fill_level * bin_capacity * revenue_per_kg
        expected_revenue = fill_levels * cap * rev
        mandatory = expected_revenue > thresh

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
