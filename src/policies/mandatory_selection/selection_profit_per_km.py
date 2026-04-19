"""
Marginal Profit-per-km Selection Module.

This module implements a routing-aware economic strategy. Unlike a pure revenue
strategy that naively selects bins based strictly on their valuable contents,
this strategy normalizes the expected revenue by the spatial cost required to
collect it. It naturally deprioritizes highly valuable but highly isolated bins,
acting as a robust proxy for marginal insertion cost.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_profit_per_km import ProfitPerKmSelection
    >>> strategy = ProfitPerKmSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy

from .base.eoq import resolve_trigger_threshold
from .base.selection_context import SelectionContext
from .base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("profit_per_km")
class ProfitPerKmSelection(IMandatorySelectionStrategy):
    """
    Economic selection strategy based on spatial ROI (Return on Investment).

    Computes a score for each bin defined as:
    Score_i = (Expected Revenue of Bin i) / (2 * Distance from Depot to Bin i)
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Selects bins whose profit-per-kilometer strictly exceeds a minimum threshold.

        Args:
            context: Selection context containing distance_matrix, fill levels,
                     bin capacity parameters, revenue multipliers, and the
                     profitability threshold.

        Returns:
            List[int]: List of bin IDs (1-based index) exceeding the profit ratio.
        """
        if context.distance_matrix is None:
            raise ValueError("ProfitPerKmSelection requires a distance_matrix.")
        if context.revenue_kg <= 0:
            # Short-circuit if there's no economic value configured
            return [], SearchContext.initialize(selection_metrics={"strategy": "ProfitPerKmSelection"})

        # Compute total mass capacity of the bin
        bin_cap = context.bin_volume * context.bin_density

        # Calculate expected revenue based on current fill ratio
        fill_ratios = context.current_fill / context.max_fill
        revenue = fill_ratios * bin_cap * context.revenue_kg

        if getattr(context, "use_eoq_threshold", False):
            # When EOQ is active, the trigger is the fill-level, but we still
            # check revenue-positivity as a secondary guard.
            mandatory_mask = resolve_trigger_threshold(context, fill_ratios)
            mandatory_indices = np.nonzero(mandatory_mask & (revenue > 0))[0]
        else:
            # distance_matrix[0] is the depot. Index 1: end are the bins.
            dist_to_depot = context.distance_matrix[0, 1:]
            # Prevent division by zero for bins that might be co-located with the depot
            safe_dist = np.where(dist_to_depot == 0, 1e-9, dist_to_depot)
            # Proxy for detour cost: Round trip distance to depot (2 * distance)
            profit_per_km = revenue / (2 * safe_dist)
            # Select bins that pass the required economic threshold
            mandatory_indices = np.nonzero(profit_per_km > context.threshold)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "ProfitPerKmSelection"}
        )
