"""
Whittle Index Selection Strategy Module.

Treats each bin as a restless armed bandit (RMAB). Each bin has two actions:
passive (let it fill) or active (collect). The Whittle index represents
the "subsidy for passivity" at which the decision-maker is indifferent
between the two actions. Bins with higher Whittle indices are prioritized.

This implementation uses a discretized MDP (20 states) and Value Iteration
to solve for the indifferent subsidy.
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("whittle")
class WhittleIndexSelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Whittle Index priority.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Rank bins by Whittle index and select top candidates.

        Args:
            context: SelectionContext with MDP parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "WhittleIndexSelection"})

        mu = context.accumulation_rates
        if mu is None:
            raise ValueError("WhittleIndexSelection requires accumulation_rates.")

        discount = context.whittle_discount
        grid_size = context.whittle_grid_size
        bin_cap = context.bin_volume * context.bin_density
        revenue_kg = context.revenue_kg
        max_fill = context.max_fill

        # Discretized MDP solve for Whittle Index
        num_states = grid_size
        states = np.linspace(0, max_fill, num_states)
        r_active = (states / max_fill) * bin_cap * revenue_kg

        whittle_indices = np.zeros(n_bins)

        for i in range(n_bins):
            mu_i = mu[i]
            # Passive transition: s -> s + mu_i (clipped to max_fill)
            # Note: searchsorted with side='right' - 1 gives nearest-from-below
            # discretization (largest grid point <= true continuous state).
            next_state_indices = np.clip(np.searchsorted(states, states + mu_i, side="right") - 1, 0, num_states - 1)
            # Clipping handles the overflow regime, pinning the bin at max_fill.

            # Binary search for the indifference subsidy m
            low_m = 0.0
            high_m = bin_cap * revenue_kg  # max possible subsidy

            # Warm-start the value function across bisection iterations
            V = np.zeros(num_states)

            # 1. State lookup is independent of m
            s_idx = int(np.clip(np.searchsorted(states, context.current_fill[i], side="right") - 1, 0, num_states - 1))

            for _b_iter in range(10):
                m = (low_m + high_m) / 2.0

                # Dynamic Programming / Value Iteration
                # Warm-start: Reuse V from previous subsidy m
                for _v_iter in range(50):
                    v_act = r_active + discount * V[0]
                    v_pass = m + discount * V[next_state_indices]

                    V_new = np.maximum(v_act, v_pass)
                    diff = np.max(np.abs(V - V_new))
                    V = V_new

                    # Ensure at least one sweep for new m before checking convergence
                    if _v_iter > 0 and diff < 1e-4:
                        break

                v_act_s = r_active[s_idx] + discount * V[0]
                v_pass_s = m + discount * V[next_state_indices[s_idx]]

                if v_act_s >= v_pass_s:
                    low_m = m
                else:
                    high_m = m

            whittle_indices[i] = (low_m + high_m) / 2.0

        # Ranking
        ranked_indices = np.argsort(-whittle_indices)

        if context.n_vehicles <= 0:
            mandatory = np.nonzero(whittle_indices > 1e-3)[0]
        else:
            mean_mass = np.mean((context.current_fill / max_fill) * bin_cap)
            mean_mass = max(mean_mass, 1e-9)
            k = int(context.n_vehicles * (context.vehicle_capacity / mean_mass))
            k = max(1, min(k, n_bins))

            top_k = ranked_indices[:k]
            mandatory = top_k[whittle_indices[top_k] > 1e-3]

        return sorted((mandatory + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "WhittleIndexSelection"}
        )
