"""
Last Minute Selection Strategy Module.

This module implements the "Last Minute" or "Reactive" strategy, which
selects bins only when they exceed a certain fill threshold (e.g., 100%).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_last_minute import LastMinuteSelection
    >>> strategy = LastMinuteSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy

from .base.eoq import resolve_trigger_threshold
from .base.selection_context import SelectionContext
from .base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("last_minute")
class LastMinuteSelection(IMandatorySelectionStrategy):
    """
    Simple threshold-based reactive strategy.

    Logic: Collect if current_fill > threshold.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins that exceed the fill threshold.

        Args:
            context: SelectionContext with fill levels and threshold.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        fill_ratios = context.current_fill / context.max_fill
        mandatory_mask = resolve_trigger_threshold(context, fill_ratios)
        mandatory_indices = np.nonzero(mandatory_mask)[0]
        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "LastMinuteSelection"}
        )
