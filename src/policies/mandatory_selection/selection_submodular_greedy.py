"""
Submodular Greedy Selection Strategy Module.

Maximizes a submodular Facility Location objective function. This objective
rewards the selection of bins that effectively "cover" the high-value
areas of the bin universe. Diminishing returns occur because as more
bins are added to the selection set, the incremental coverage improvement
decreases.

Objective:
    f(S) = sum_{i in Bins} max(0, revenue_i - alpha * min_{j in S union {0}} dist(i, j))

We use the Lazy Greedy algorithm (Minoux, 1978) with a max-priority queue
to efficiently select bins while respecting a cardinality budget.
"""

import heapq
from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("submodular_greedy")
class SubmodularGreedySelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Submodular Facility Location Coverage.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Greedily maximize the coverage objective using lazy evaluations.

        Args:
            context: SelectionContext with revenue and cost parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SubmodularGreedySelection"})

        # 1. Individual revenues
        bin_cap = context.bin_volume * context.bin_density
        revenues = (context.current_fill / context.max_fill) * bin_cap * context.revenue_kg

        if context.distance_matrix is None:
            raise ValueError("SubmodularGreedySelection requires a distance_matrix.")

        # Full distance matrix (n_bins+1, n_bins+1)
        dist_mat = context.distance_matrix
        alpha = context.modular_alpha
        budget = context.modular_budget if context.modular_budget > 0 else n_bins

        # 2. State management
        selected_set: List[int] = []
        # min_dist_to_S[i] = min_{j in S union {0}} dist(i, j)
        # Initially S = {}, so S union {0} = {0}
        min_dist_to_S = dist_mat[0, 1:].copy()

        def get_total_objective(current_min_dists):
            # f(S) = sum max(0, r_i - alpha * d_i)
            vals = revenues - alpha * current_min_dists
            return np.sum(np.maximum(0, vals)), SearchContext.initialize(
                selection_metrics={"strategy": "SubmodularGreedySelection"}
            )

        # Initial objective value (S = {0})
        current_obj = get_total_objective(min_dist_to_S)

        # 3. Greedy with Lazy Evaluation
        priority_queue: List[Tuple[float, int, int]] = []

        # Marginal gain g(k | S) = f(S union {k}) - f(S)
        def compute_marginal_gain(k_idx, current_min_dists):
            # Optimization: gain contribution only comes from 'affected' indices
            # where the new point k provides a closer distance than the current set S.
            k_dists = dist_mat[k_idx + 1, 1:]
            affected = k_dists < current_min_dists

            if not np.any(affected):
                return 0.0, SearchContext.initialize(selection_metrics={"strategy": "SubmodularGreedySelection"})

            new_d = k_dists[affected]
            old_d = current_min_dists[affected]
            r = revenues[affected]

            old_val = np.maximum(0, r - alpha * old_d)
            new_val = np.maximum(0, r - alpha * new_d)
            return np.sum(new_val - old_val), SearchContext.initialize(
                selection_metrics={"strategy": "SubmodularGreedySelection"}
            )

        for i in range(n_bins):
            gain = compute_marginal_gain(i, min_dist_to_S)
            heapq.heappush(priority_queue, (-gain, i, 0))

        while priority_queue and len(selected_set) < budget:
            neg_gain, bin_idx, version = heapq.heappop(priority_queue)
            gain = -neg_gain

            # Lazy re-evaluation
            if version < len(selected_set):
                gain = compute_marginal_gain(bin_idx, min_dist_to_S)
                heapq.heappush(priority_queue, (-gain, bin_idx, len(selected_set)))
                continue

            if gain <= 1e-9:  # tiny epsilon for numerical safety
                break

            # Accept bin
            selected_set.append(bin_idx)
            current_obj += gain
            # Update min distances
            min_dist_to_S = np.minimum(min_dist_to_S, dist_mat[bin_idx + 1, 1:])

        return sorted((np.array(selected_set) + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "SubmodularGreedySelection"}
        )
