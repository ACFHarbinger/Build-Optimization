"""
Pareto-Front Selection Strategy Module.

This module implements a sophisticated combinatorial selection strategy based on
multi-objective optimization. Instead of relying on a single scalar score, it
evaluates bins on a bi-objective Pareto front:
    1. Urgency: Minimizing the expected days until overflow.
    2. Routing Efficiency: Minimizing the spatial distance to the depot (as a
       proxy for insertion cost).

Any bin that lies on the non-dominated Pareto front is strictly obligated for
collection. This ensures that the policy never leaves behind a bin that is either
critically urgent or trivially cheap to collect without a justified trade-off.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_pareto import ParetoFrontSelection
    >>> strategy = ParetoFrontSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("pareto_front")
class ParetoFrontSelection(IMandatorySelectionStrategy):
    """
    Combinatorial selection strategy utilizing non-dominated sorting.

    Evaluates the set of all bins and selects the subset of non-dominated bins
    in the 2D objective space (Days to Overflow $\\times$ Distance to Depot).
    A bin $v_i$ dominates $v_j$ if it is strictly more urgent AND closer to the depot.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Selects the non-dominated set of bins based on urgency and routing cost.

        A pre-filtering step is applied using `context.threshold` to discard bins
        that have an abundance of remaining time (e.g., > 7 days). This prevents
        degenerate Pareto solutions where an empty bin is selected simply because
        it happens to be physically adjacent to the depot.

        Args:
            context: Selection context containing current fill levels,
                     accumulation rates, distance_matrix, and an urgency threshold.

        Returns:
            List[int]: List of non-dominated bin IDs (1-based index).
        """
        if context.distance_matrix is None:
            raise ValueError("ParetoFrontSelection requires a distance_matrix in the context.")
        if context.accumulation_rates is None:
            raise ValueError("ParetoFrontSelection requires accumulation_rates in the context.")

        rem_capacity = context.max_fill - context.current_fill

        # Prevent division by zero for bins with zero expected accumulation
        mu = np.where(context.accumulation_rates == 0, 1e-9, context.accumulation_rates)

        # Objective 1: Urgency (Minimize days to overflow)
        days_to_overflow = rem_capacity / mu

        # Objective 2: Routing Cost Proxy (Minimize distance to depot)
        # Note: distance_matrix[0] is the depot; [1:] maps to the bins.
        dist_to_depot = context.distance_matrix[0, 1:]

        mandatory = []
        n_bins = len(days_to_overflow)

        for i in range(n_bins):
            # Pre-filter: Do not evaluate bins that are comfortably far from overflowing.
            # context.threshold acts as the maximum acceptable lookahead horizon here.
            if days_to_overflow[i] > context.threshold:
                continue

            dominated = False
            for j in range(n_bins):
                if i == j:
                    continue

                # Check for strict Pareto dominance.
                # Bin j dominates Bin i if it is better or equal in both objectives,
                # and strictly better in at least one objective.
                if (days_to_overflow[j] <= days_to_overflow[i] and dist_to_depot[j] <= dist_to_depot[i]) and (
                    days_to_overflow[j] < days_to_overflow[i] or dist_to_depot[j] < dist_to_depot[i]
                ):
                    dominated = True
                    break

            if not dominated:
                mandatory.append(i + 1)

        return mandatory, SearchContext.initialize(selection_metrics={"strategy": "ParetoFrontSelection"})
