"""
Deadline-Driven Selection Strategy Module.

This strategy computes the exact floor of days remaining until guaranteed
overflow based on the expected accumulation rate. It obligates collection
for any bin whose deadline is less than or equal to a defined lookahead horizon.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_deadline import DeadlineDrivenSelection
    >>> strategy = DeadlineDrivenSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("deadline")
class DeadlineDrivenSelection(IMandatorySelectionStrategy):
    """
    Temporal selection strategy based on exact day-to-overflow calculation.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Selects bins that will reach maximum capacity within the lookahead horizon.

        Args:
            context: Selection context containing current fill levels,
                     accumulation rates, and horizon thresholds.

        Returns:
            List[int]: List of bin IDs (1-based index) strictly required for collection.
        """
        if context.accumulation_rates is None:
            raise ValueError("DeadlineDrivenSelection requires accumulation_rates.")

        # Extract horizon dynamically, defaulting to context.threshold for backwards compatibility
        horizon_days = getattr(context, "horizon_days", context.threshold)

        rem_capacity = context.max_fill - context.current_fill

        # Prevent division by zero for bins with zero expected accumulation
        mu = np.where(context.accumulation_rates == 0, 1e-9, context.accumulation_rates)

        # Compute the expected number of days until the bin hits capacity
        # Equation: d* = floor((tau - w) / mu)
        days_to_overflow = np.floor(rem_capacity / mu)

        # Select bins whose deadline is within the requested horizon
        mandatory_indices = np.nonzero(days_to_overflow <= horizon_days)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "DeadlineDrivenSelection"}
        )
