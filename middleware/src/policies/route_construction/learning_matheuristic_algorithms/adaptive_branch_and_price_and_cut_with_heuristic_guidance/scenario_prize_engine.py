from __future__ import annotations

from typing import Dict

import numpy as np

# Corrected imports as found in the repository
from logic.src.pipeline.simulations.bins.prediction import (
    ScenarioTree,
    calculate_frequency_and_level,
    predict_days_to_overflow,
)


class ScenarioPrizeEngine:
    """
    Computes multi-period augmented node prizes from a ScenarioTree.

    For each bin i, the augmented prize is:
        π_i = w_{i,d} * R                     ← immediate revenue
            + γ * E_ξ[V_future(i, not visited)] ← opportunity cost of deferral
            - γ * E_ξ[V_future(i, visited)]     ← value of visiting now

    Where V_future is approximated analytically from the scenario tree
    using the Gamma-distribution overflow predictor.
    """

    def __init__(
        self,
        scenario_tree: ScenarioTree,
        gamma: float = 0.95,  # inter-day discount
        tau: float = 100.0,  # max bin capacity
        overflow_penalty: float = 2.0,  # revenue multiplier for overflow risk
    ):
        self.tree = scenario_tree
        self.gamma = gamma
        self.tau = tau
        self.overflow_penalty = overflow_penalty

    def compute_prizes(
        self,
        current_wastes: np.ndarray,  # w_{i,d}, shape (n_bins,)
        bin_stats: Dict[str, np.ndarray],  # {"means": μ_i, "stds": σ_i}
        revenue: float,
        days_remaining: int,
    ) -> Dict[int, float]:
        """
        Compute scenario-augmented node prizes for all bins.

        Returns:
            Dict mapping bin index (1-based) to augmented prize.
        """
        n_bins = len(current_wastes)
        means = bin_stats["means"]  # E[δ_i]
        stds = bin_stats.get("stds", np.ones(n_bins))
        variances = stds**2

        # 1. Days-to-overflow per bin using existing prediction function
        days_to_overflow = predict_days_to_overflow(
            ui=means,
            vi=variances,
            f=current_wastes,
            cl=0.9,  # 90% service level
        )

        prizes: Dict[int, float] = {}

        for i in range(n_bins):
            bin_id = i + 1

            # Immediate revenue
            base = float(current_wastes[i]) * revenue

            # ── Future value if NOT visited today ─────────────────────────
            v_leave = self._expected_future_value_leave(
                idx=i,
                current_fill=current_wastes[i],
                mean_rate=means[i],
                variance=variances[i],
                days_to_overflow=days_to_overflow[i],
                days_remaining=days_remaining,
                revenue=revenue,
            )

            # ── Future value if visited today ─────────────────────────────
            v_visit = self._expected_future_value_visit(
                idx=i,
                mean_rate=means[i],
                variance=variances[i],
                days_remaining=days_remaining,
                revenue=revenue,
            )

            # Augmented prize = immediate + discounted opportunity gain
            rho = self.gamma * (v_leave - v_visit)
            prizes[bin_id] = base + rho

        return prizes

    def _expected_future_value_leave(
        self,
        idx: int,
        current_fill: float,
        mean_rate: float,
        variance: float,
        days_to_overflow: float,
        days_remaining: int,
        revenue: float,
    ) -> float:
        """
        E[future value | bin not visited today].

        If days_to_overflow < days_remaining: overflow risk applies.
        Otherwise: expected accumulated revenue at next optimal visit.
        """
        if days_to_overflow <= 1.0:
            # Will overflow before next day — certain penalty
            return -self.overflow_penalty * self.tau * revenue

        # Expected fill at next visit (capped at τ)
        expected_fill_next = min(current_fill + mean_rate * min(days_to_overflow, days_remaining), self.tau)
        # Discounted revenue at next visit
        discount = self.gamma ** int(min(days_to_overflow, days_remaining))
        return discount * expected_fill_next * revenue

    def _expected_future_value_visit(
        self,
        idx: int,
        mean_rate: float,
        variance: float,
        days_remaining: int,
        revenue: float,
    ) -> float:
        """
        E[future value | bin visited today — resets to ~mean_rate fill].

        Bin accumulates from reset; next visit at optimal frequency.
        """
        opt_freq, target_level = calculate_frequency_and_level(ui=mean_rate, vi=variance, cf=0.9)

        if opt_freq >= days_remaining:
            # Won't be worth visiting again before horizon ends
            return 0.0

        discount = self.gamma**opt_freq
        return discount * target_level * revenue

    def scenario_weighted_prizes(
        self,
        day: int,
        revenue: float,
        days_remaining: int,
    ) -> Dict[int, float]:
        """
        Aggregate prizes across all scenarios at a given day,
        weighted by scenario probability.
        """
        scenarios_at_day = self.tree.get_scenarios_at_day(day)
        if not scenarios_at_day:
            return {}

        # Extract stats from scenario distribution at this day

        # Extract stats from scenario distribution at this day
        all_wastes = np.stack([s.wastes for s in scenarios_at_day])
        probs = np.array([s.probability for s in scenarios_at_day])
        probs /= probs.sum()  # normalize

        # Scenario-weighted mean and variance of fills
        means_fill = np.average(all_wastes, axis=0, weights=probs)
        vars_fill = np.average((all_wastes - means_fill) ** 2, axis=0, weights=probs)

        # We assume wastes accumulated steadily.
        # Calculate mean rates conceptually from differences if possible.
        previous_scenarios = self.tree.get_scenarios_at_day(day - 1) if day > 0 else []
        if previous_scenarios:
            prev_all = np.stack([s.wastes for s in previous_scenarios])
            prev_probs = np.array([s.probability for s in previous_scenarios])
            prev_probs /= prev_probs.sum()
            prev_means = np.average(prev_all, axis=0, weights=prev_probs)
        else:
            prev_means = means_fill  # Fallback if no prev day exists, though usually day>0 implies it does

        bin_stats = {
            "means": np.maximum(means_fill - prev_means, 0.0),
            "stds": np.sqrt(vars_fill),
        }

        return self.compute_prizes(
            current_wastes=means_fill,
            bin_stats=bin_stats,
            revenue=revenue,
            days_remaining=days_remaining,
        )
