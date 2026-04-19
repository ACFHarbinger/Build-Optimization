"""
Service Level (Statistical) Selection Strategy.
"""

from typing import Optional

from torch import Tensor

from .base import VectorizedSelector


class ServiceLevelSelector(VectorizedSelector):
    """
    Statistical overflow prediction strategy.

    Uses mean accumulation rate and standard deviation to predict
    overflow probability and select bins accordingly.
    """

    def __init__(self, confidence_factor: float = 1.0, max_fill: float = 1.0):
        """
        Initialize ServiceLevelSelector.

        Args:
            confidence_factor: Number of standard deviations for prediction.
                               Higher = more conservative (fewer overflows).
            max_fill: Maximum fill level (overflow threshold). Default: 1.0.
        """
        self.confidence_factor = confidence_factor
        self.max_fill = max_fill

    def select(
        self,
        fill_levels: Tensor,
        accumulation_rates: Optional[Tensor] = None,
        std_deviations: Optional[Tensor] = None,
        confidence_factor: Optional[float] = None,
        max_fill: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins statistically likely to overflow.

        Prediction: current + rate + (confidence * std) >= max_fill

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            accumulation_rates: Mean daily fill rate (batch_size, num_nodes).
            std_deviations: Standard deviation of fill rate (batch_size, num_nodes).
            confidence_factor: Optional override for confidence multiplier.
            max_fill: Optional override for overflow threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        conf = confidence_factor if confidence_factor is not None else self.confidence_factor
        overflow_thresh = max_fill if max_fill is not None else self.max_fill

        if accumulation_rates is None or std_deviations is None:
            # Without statistics, fall back to threshold-based
            mandatory = fill_levels >= overflow_thresh
        else:
            # Statistical prediction: current + mean + confidence * std
            predicted_fill = fill_levels + accumulation_rates + (conf * std_deviations)
            mandatory = predicted_fill >= overflow_thresh

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
