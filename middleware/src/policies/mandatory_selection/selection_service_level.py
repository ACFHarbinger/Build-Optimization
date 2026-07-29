"""
Service Level Selection Strategy Module.

This module implements a robust optimization baseline that selects bins
to maintain a service level guarantee. It uses a linear projection of
the expected accumulation and standard deviation over a defined horizon
to ensure bins are collected before their worst-case prediction hits capacity.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_service_level import ServiceLevelSelection
    >>> strategy = ServiceLevelSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("service_level")
class ServiceLevelSelection(IMandatorySelectionStrategy):
    """
    Statistical overflow prediction strategy using linear confidence bounds.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins that are statistically likely to overflow within the horizon.

        Args:
            context: Selection context containing fill levels, accumulation_rates,
                     std_deviations, confidence threshold, and horizon_days.

        Returns:
            List[int]: List of bin IDs (1-based) predicted to overflow.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            return [], SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})

        # Extract horizon parameter dynamically (defaults to 1 for next-day projection)
        horizon_days = getattr(context, "horizon_days", 1)

        # Linear projection of worst-case fill over the horizon:
        # w_future = w_current + (D * mu) + (D * k * sigma)
        predicted_fill = (
            context.current_fill
            + (context.accumulation_rates * horizon_days)
            + (context.threshold * context.std_deviations * horizon_days)
        )

        # Identify bins where the worst-case projection exceeds capacity
        mandatory = np.nonzero(predicted_fill >= MAX_CAPACITY_PERCENT)[0] + 1
        return mandatory.tolist(), SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})
