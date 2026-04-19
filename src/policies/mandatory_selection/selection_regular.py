"""
Regular Selection Strategy Module.

This module implements a fixed-frequency selection strategy. Bins are selected
if the current day matches their assigned schedule frequency (e.g., every 3 days).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_regular import RegularSelection
    >>> strategy = RegularSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import (
    MandatorySelectionRegistry,
)


@MandatorySelectionRegistry.register("regular")
class RegularSelection(IMandatorySelectionStrategy):
    """
    Periodic collection strategy.

    Logic: Collect all bins if today is a scheduled collection day.
    Scheduled if (current_day - 1 % frequency) == 0.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select all bins if it is a scheduled collection day.

        Args:
            context: SelectionContext with day and threshold (freq).

        Returns:
            List[int]: List of all bin IDs if collection day, else empty.
        """
        current_day = getattr(context, "current_day", 1)
        # We want to run on day 1, and then on day X+1, 2X+1, etc.
        # where X is the frequency (threshold).
        threshold = getattr(context, "threshold", 1)  # Using threshold as frequency
        if (current_day - 1) % threshold != 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})

        # Find bins where current fill >= min_fill
        eligible_bins = []
        if hasattr(context, "current_fill") and context.current_fill is not None and len(context.current_fill) > 0:
            min_fill = getattr(self, "min_fill", 0)
            for i, fill in enumerate(context.current_fill):
                if fill >= min_fill:
                    eligible_bins.append(i + 1)  # 1-based indexing for routing
        else:
            eligible_bins = (context.bin_ids + 1).tolist() if hasattr(context, "bin_ids") else []
        return eligible_bins, SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})
