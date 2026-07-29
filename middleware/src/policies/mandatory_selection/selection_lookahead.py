"""
Lookahead Selection Strategy Module.

This module implements a predictive strategy that selects bins based on
future projections. It simulates future days to identify critical bins that
will overflow soon or are efficient to collect now.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_lookahead import LookaheadSelection
    >>> strategy = LookaheadSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("lookahead")
class LookaheadSelection(IMandatorySelectionStrategy):
    """
    Predictive selection strategy looking ahead to synchronize collections.

    Selects bins that will overflow before the next required visit to currently full bins.
    """

    def _should_bin_be_collected(self, current_fill_level: float, accumulation_rate: float) -> bool:
        """
        Check if bin overflows today.
        """
        return current_fill_level + accumulation_rate >= MAX_CAPACITY_PERCENT

    def _update_fill_levels_after_first_collection(
        self, bin_indices: List[int], mandatory_bins: List[int], current_fill_levels: np.ndarray
    ) -> np.ndarray:
        """
        Simulate collection for mandatory_bins by setting fill levels to 0.
        """
        for i in bin_indices:
            if i in mandatory_bins:
                current_fill_levels[i] = 0
        return current_fill_levels

    def _initialize_lists_bins(self, n_bins: int) -> List[int]:
        """
        Initialize list for next collection days.
        """
        return [0] * n_bins

    def _calculate_next_collection_days(
        self,
        bin_indices: List[int],
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
    ) -> List[int]:
        """
        Calculate when collected bins would overflow again.
        """
        next_collection_days = self._initialize_lists_bins(len(bin_indices))
        # Work on a copy to avoid side effects
        temporary_fill_levels = current_fill_levels.copy()
        for i in mandatory_bins:
            current_day = 0
            rate = accumulation_rates[i]
            if rate <= 0:
                continue

            while temporary_fill_levels[i] < MAX_CAPACITY_PERCENT:
                temporary_fill_levels[i] = temporary_fill_levels[i] + rate
                current_day = current_day + 1

            next_collection_days[i] = current_day  # assuming collection happens at the beginning of the day
        return next_collection_days

    def _get_next_collection_day(
        self,
        bin_indices: List[int],
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
    ) -> int:
        """
        Find the earliest overflow day among the currently selected bins.
        """
        next_collection_days = self._calculate_next_collection_days(
            bin_indices, mandatory_bins, current_fill_levels, accumulation_rates
        )
        next_collection_days_array = np.array(next_collection_days)
        # Find minimum non-zero day
        non_zero_indices = np.nonzero(next_collection_days_array)
        if len(non_zero_indices[0]) == 0:
            return 0

        next_collection_day = np.min(next_collection_days_array[non_zero_indices])
        return int(next_collection_day)

    def _add_bins_to_collect(
        self,
        bin_indices: List[int],
        next_collection_day: int,
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
    ) -> List[int]:
        """
        Add bins that would overflow before the next collection day.
        """
        # Assuming current_collection_day is 0 (relative start)
        current_collection_day = 0
        for i in bin_indices:
            if i in mandatory_bins:
                continue
            else:
                for j in range(current_collection_day + 1, next_collection_day):
                    if current_fill_levels[i] + j * accumulation_rates[i] >= MAX_CAPACITY_PERCENT:
                        mandatory_bins.append(i)
                        break
        return mandatory_bins

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins based on lookahead logic.
        """
        if context.accumulation_rates is None:
            return [], SearchContext.initialize(selection_metrics={"strategy": "LookaheadSelection"})

        current_fill_levels = context.current_fill
        accumulation_rates = context.accumulation_rates
        n_bins = len(current_fill_levels)
        bin_indices = list(range(n_bins))

        mandatory_bins = []
        for i in bin_indices:
            if self._should_bin_be_collected(current_fill_levels[i], accumulation_rates[i]):
                mandatory_bins.append(i)

        if mandatory_bins:
            # Create a copy for simulation
            simulated_fill_levels = self._update_fill_levels_after_first_collection(
                bin_indices, mandatory_bins, current_fill_levels.copy()
            )

            next_collection_day = self._get_next_collection_day(
                bin_indices, mandatory_bins, simulated_fill_levels, accumulation_rates
            )

            if next_collection_day > 0:
                mandatory_bins = self._add_bins_to_collect(
                    bin_indices,
                    next_collection_day,
                    mandatory_bins,
                    current_fill_levels,  # Original fill levels
                    accumulation_rates,
                )

        # Convert back to 1-based IDs
        return [i + 1 for i in mandatory_bins], SearchContext.initialize(
            selection_metrics={"strategy": "LookaheadSelection"}
        )
