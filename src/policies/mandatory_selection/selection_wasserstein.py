"""
Wasserstein Robust Selection Strategy Module.

Implements a distributionally robust selection strategy using a Wasserstein-1
ambiguity ball of radius epsilon around the empirical Gaussian distribution.
Instead of optimizing for the expected overflow under a single point estimate,
it considers the worst-case expectation over all distributions within the
Wasserstein-1 distance. Utilizing the duality result from Mohajerin Esfahani
& Kuhn (2018), this worst-case expectation for the ReLU/Max loss is simply
the nominal expectation plus the radius epsilon.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_wasserstein import WassersteinRobustSelection
    >>> strategy = WassersteinRobustSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("wasserstein_robust")
class WassersteinRobustSelection(IMandatorySelectionStrategy):
    """
    Distributionally robust selection strategy based on Wasserstein ambiguity balls.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins based on the worst-case expected overflow volume.

        Args:
            context: SelectionContext with Wasserstein radius and Gaussian params.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if context.wasserstein_p != 1:
            raise NotImplementedError("WassersteinRobustSelection currently only supports p=1.")

        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "WassersteinRobustSelection"})

        mu = context.accumulation_rates
        if mu is None:
            raise ValueError("WassersteinRobustSelection requires accumulation_rates.")

        sigma = context.std_deviations if context.std_deviations is not None else np.zeros(n_bins)
        epsilon = context.wasserstein_radius
        max_fill = context.max_fill
        threshold = context.threshold

        # Step 1: Compute Nominal Expected Overflow E_P[max(0, F - max_fill)]
        # F ~ N(current_fill + mu, sigma^2)
        x_mu = (context.current_fill + mu) - max_fill
        x_sigma = np.where(sigma <= 1e-9, 1e-9, sigma)

        # Expected value of max(0, X) where X ~ N(mu_x, sigma_x^2)
        # E[max(0, X)] = mu_x * Phi(mu_x / sigma_x) + sigma_x * phi(mu_x / sigma_x)
        z = x_mu / x_sigma
        nominal_expectation = x_mu * norm.cdf(z) + x_sigma * norm.pdf(z)

        # For deterministic bins
        deterministic_overflow = np.maximum(0, x_mu)
        nominal_expectation = np.where(sigma <= 1e-9, deterministic_overflow, nominal_expectation)

        # Step 2: Add Wasserstein-1 robust adjustment
        # Worst-case expected loss = nominal_expectation + epsilon * Lipschitz_constant
        # For max(0, x), the Lipschitz constant is 1.
        robust_expectation = nominal_expectation + epsilon

        mandatory_indices = np.nonzero(robust_expectation > threshold)[0]

        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "WassersteinRobustSelection"}
        )
