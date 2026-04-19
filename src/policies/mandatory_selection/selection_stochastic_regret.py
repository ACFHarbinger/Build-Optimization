"""
Stochastic Regret Selection Strategy Module.

This module implements the Expected Overflow Regret strategy. It calculates
the mathematically exact expected overflow volume of a bin if its collection
is deferred to the next day, assuming normally distributed accumulation rates.
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("stochastic_regret")
class StochasticRegretSelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Expected Overflow Regret (EOR).

    Selects bins whose expected overflow volume for the next day
    exceeds a defined threshold.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins based on the mathematical expectation of their overflow.

        Args:
            context: SelectionContext containing current fill, mu, sigma, and threshold.

        Returns:
            List[int]: List of bin IDs (1-based index) strictly required for collection.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            raise ValueError("StochasticRegretSelection requires both accumulation_rates and std_deviations.")

        # Remaining capacity
        c = context.max_fill - context.current_fill
        mu = context.accumulation_rates
        sigma = context.std_deviations

        # Prevent division by zero for deterministic bins
        safe_sigma = np.where(sigma == 0, 1e-9, sigma)

        # Z-score for the remaining capacity
        Z = (c - mu) / safe_sigma

        # Closed-form expected value of max(0, X - c) for X ~ N(mu, sigma^2)
        expected_overflow = sigma * norm.pdf(Z) + (mu - c) * (1 - norm.cdf(Z))

        # Fallback for purely deterministic edge cases
        expected_overflow = np.where(sigma == 0, np.maximum(0, mu - c), expected_overflow)

        # Select bins where the expected overflow exceeds the acceptable threshold
        # (context.threshold here acts as the parameter \gamma from the paper)
        mandatory_indices = np.nonzero(expected_overflow > context.threshold)[0]

        # Return 1-based indexing for the routing engine
        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "StochasticRegretSelection"}
        )
