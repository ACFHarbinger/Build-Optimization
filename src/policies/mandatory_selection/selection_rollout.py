"""
Rollout Selection Strategy Module.

Implements a one-step rollout algorithm that evaluates the expected future
reward of "collecting today" versus "deferring to tomorrow" for each bin.
The lookahead is performed using a simple simulation over a fixed horizon
under a base policy (e.g., last-minute collection).

Note:
    This is an approximation and should be documented as such in research.
    Currently, the lookahead evaluates each bin in isolation rather than
    simulating the full vehicle fleet and routing across all potential nodes.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_rollout import RolloutSelection
    >>> strategy = RolloutSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("rollout")
class RolloutSelection(IMandatorySelectionStrategy):
    """
    Lookahead-based selection strategy using one-step rollout.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins by comparing simulated future rewards.

        Args:
            context: SelectionContext with simulation parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        # Lazy import to avoid circular dependencies
        from logic.src.policies.mandatory_selection.base.selection_factory import MandatorySelectionFactory

        # TODO: Implement full-system rollout that simulates the entire state
        # (all bins) and uses the base policy on the full context to capture
        # multi-bin routing synergies and capacity constraints.

        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "RolloutSelection"})

        mu = context.accumulation_rates
        if mu is None:
            raise ValueError("RolloutSelection requires accumulation_rates.")

        sigma = context.std_deviations if context.std_deviations is not None else np.zeros(n_bins)
        horizon = context.rollout_horizon
        n_scenarios = context.rollout_n_scenarios
        base_policy_name = context.rollout_base_policy or "last_minute"

        try:
            base_policy = MandatorySelectionFactory.create_strategy(base_policy_name)
        except Exception:
            # Fallback if policy creation fails
            from logic.src.policies.mandatory_selection.selection_last_minute import LastMinuteSelection

            base_policy = LastMinuteSelection()

        bin_cap = context.bin_volume * context.bin_density
        revenue_kg = context.revenue_kg
        cost_per_km = context.cost_per_km
        max_fill = context.max_fill

        dist_to_depot = context.distance_matrix[0, 1:] if context.distance_matrix is not None else np.zeros(n_bins)
        round_trip_cost = 2 * dist_to_depot * cost_per_km

        mandatory_indices = []

        for i in range(n_bins):
            # Evaluate "Collect Today" vs "Defer" for bin i
            r_collect = self._eval_bin(
                i,
                True,
                context,
                base_policy,
                horizon,
                n_scenarios,
                bin_cap,
                revenue_kg,
                round_trip_cost,
                max_fill,
                sigma[i],
            )
            r_defer = self._eval_bin(
                i,
                False,
                context,
                base_policy,
                horizon,
                n_scenarios,
                bin_cap,
                revenue_kg,
                round_trip_cost,
                max_fill,
                sigma[i],
            )

            if r_collect > r_defer:
                mandatory_indices.append(i)

        return sorted((np.array(mandatory_indices) + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "RolloutSelection"}
        )

    def _eval_bin(
        self,
        idx: int,
        collect_today: bool,
        context: SelectionContext,
        base_policy: IMandatorySelectionStrategy,
        horizon: int,
        n_scenarios: int,
        bin_cap: float,
        revenue_kg: float,
        round_trip_cost: np.ndarray,
        max_fill: float,
        target_sigma: float,
    ) -> float:
        """
        Evaluate expected reward for a single bin.

        Note:
            This is an approximation using an isolated single-bin simulation.
            Synergies and capacity constraints from other bins are not modeled here.
            TODO: Implement full-system rollout that simulates the entire state.

            This evaluation assumes cost_per_km > 0 to resolve ties between
            "collect today" and "defer indefinitely" (both zeros if costs=0).
        """
        mu = context.accumulation_rates[idx] if context.accumulation_rates is not None else 0.0
        sigma = target_sigma
        # Dedicated rollout discount with fallback to whittle_discount
        discount = getattr(context, "rollout_discount", getattr(context, "whittle_discount", 0.95))

        total_reward = 0.0
        current_fill = context.current_fill[idx]

        # Hoist mock context construction to avoid allocations in the scenario loop
        mock_ctx = SelectionContext(
            bin_ids=np.array([idx], dtype=np.int32),
            current_fill=np.zeros(1, dtype=np.float64),
            threshold=context.threshold,
            bin_volume=context.bin_volume,
            bin_density=context.bin_density,
            revenue_kg=revenue_kg,
            max_fill=max_fill,
        )

        for _s in range(n_scenarios):
            scenario_reward = 0.0
            current_f = current_fill

            # Step 1: Decision for TODAY
            if collect_today:
                revenue = (current_f / max_fill) * bin_cap * revenue_kg
                scenario_reward += revenue - round_trip_cost[idx]
                current_f = 0.0

            # Step 2: Roll out for horizon days
            for t in range(1, horizon + 1):
                # Accumulate
                delta = np.random.normal(mu, sigma) if n_scenarios > 1 and sigma > 0 else mu
                current_f += delta

                # Apply discount factor to future rewards/penalties
                discount_t = discount**t

                # Check for overflow penalty (soft penalty: 10x revenue)
                if current_f > max_fill:
                    overflow = current_f - max_fill
                    scenario_reward -= discount_t * (overflow / max_fill) * bin_cap * revenue_kg * 10.0

                # Apply base policy to decide if we collect tomorrow
                mock_ctx.current_fill[0] = current_f
                if base_policy.select_bins(mock_ctx):
                    revenue = (current_f / max_fill) * bin_cap * revenue_kg
                    # Consistently discount both revenue and costs at time t
                    scenario_reward += discount_t * (revenue - round_trip_cost[idx])
                    current_f = 0.0

            total_reward += scenario_reward

        return total_reward / n_scenarios
