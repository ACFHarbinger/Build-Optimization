"""
Multi-Day Overflow Probability Selection Module.

This strategy evaluates the tail probability of a bin overflowing over a
stochastic horizon of K days. Assuming daily waste generation is i.i.d Gaussian,
the total accumulation variance scales by the square root of time (sqrt(D)),
providing a much more mathematically rigorous risk assessment than linear bounds.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_multi_day_prob import MultiDayOverflowSelection
    >>> strategy = MultiDayOverflowSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("multi_day_prob")
class MultiDayOverflowSelection(IMandatorySelectionStrategy):
    """
    Stochastic selection strategy based on cumulative probability over time.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Selects bins if their probability of overflowing within K days
        exceeds a defined risk threshold.

        Args:
            context: Selection context containing stochastic parameters
                     and optionally `horizon_days`.

        Returns:
            List[int]: List of bin IDs (1-based index) exceeding risk thresholds.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            raise ValueError("Missing stochastic parameters in context.")

        # Extract horizon parameter dynamically
        horizon_days = getattr(context, "horizon_days", 3)

        rem_capacity = context.max_fill - context.current_fill

        # Scale parameters for K days. Variance scales linearly with time,
        # meaning standard deviation scales with the square root of time.
        mu_k = context.accumulation_rates * horizon_days
        sigma_k = context.std_deviations * np.sqrt(horizon_days)

        # Prevent division by zero for deterministic cases
        safe_sigma = np.where(sigma_k == 0, 1e-9, sigma_k)

        # Compute the probability of the accumulation exceeding remaining capacity
        # P(Accumulation >= Remaining Capacity)
        Z = (rem_capacity - mu_k) / safe_sigma
        prob_overflow = 1.0 - norm.cdf(Z)

        # Deterministic fallback for bins with 0 variance
        prob_overflow = np.where(sigma_k == 0, (mu_k >= rem_capacity).astype(float), prob_overflow)

        # Here context.threshold acts as the maximum acceptable risk probability (e.g., 0.85)
        mandatory_indices = np.nonzero(prob_overflow >= context.threshold)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "MultiDayOverflowSelection"}
        )
