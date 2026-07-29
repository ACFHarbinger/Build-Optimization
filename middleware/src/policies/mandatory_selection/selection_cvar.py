"""
CVaR Selection Strategy Module.

Computes the Conditional Value-at-Risk (CVaR) at level alpha for the anticipated
overflow volume in the next period. Under a Gaussian assumption for accumulation,
this risk measure provides a robust estimate of "tail" overflow risk, focusing
on extreme outcomes rather than just the mean.
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("cvar")
class CVaRSelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Tail risk (CVaR of overflow).
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins where tail overflow risk exceeds the threshold.

        Args:
            context: SelectionContext with Gaussian parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "CVaRSelection"})

        mu = context.accumulation_rates
        if mu is None:
            raise ValueError("CVaRSelection requires accumulation_rates.")

        sigma = context.std_deviations if context.std_deviations is not None else np.zeros(n_bins)
        alpha = context.cvar_alpha
        max_fill = context.max_fill
        threshold = context.threshold

        # Future fill: F ~ N(current_fill + mu, sigma^2)
        # Surplus: X = F - max_fill ~ N(current_fill + mu - max_fill, sigma^2)
        x_mu = context.current_fill + mu - max_fill
        x_sigma = np.where(sigma <= 1e-9, 1e-9, sigma)

        # VaR_alpha for X
        z_alpha = norm.ppf(alpha)
        var_alpha = x_mu + x_sigma * z_alpha

        # CVaR_alpha for X (signed surplus)
        cvar_x = x_mu + x_sigma * (norm.pdf(z_alpha) / (1.0 - alpha))

        # We care about CVaR_alpha(max(0, X)) i.e. the tail of overflow.
        # 1. If var_alpha > 0, the entire tail is positive, so CVaR(max(0, X)) = CVaR(X).
        # 2. If var_alpha <= 0, the tail includes 0.
        # Approximation when var_alpha <= 0: we use E[max(0,X)] / (1 - alpha) as an
        # upper bound on CVaR_alpha(max(0,X)). Tight when P(X > 0) <= 1 - alpha.

        # Expected value E[max(0, X)]
        z_zero = x_mu / x_sigma
        expected_overflow = x_mu * norm.cdf(z_zero) + x_sigma * norm.pdf(z_zero)

        # Handle deterministic case
        deterministic_overflow = np.maximum(0, x_mu)

        # Selection criterion
        if np.all(sigma <= 1e-9):
            final_risk = deterministic_overflow
        else:
            # Vectorized selection: check if tail starts above zero
            final_risk = np.where(var_alpha > 0, cvar_x, expected_overflow / (1.0 - alpha))

        mandatory_indices = np.nonzero(final_risk > threshold)[0]

        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "CVaRSelection"}
        )
