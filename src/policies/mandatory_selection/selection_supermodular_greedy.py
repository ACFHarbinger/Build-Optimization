r"""
Supermodular Greedy Selection Strategy Module.

Maximizes a synergetic objective function that balances expected revenue
against a lower bound on the TSP route length. The objective exhibits
increasing returns (supermodularity) because adding a bin to a cluster
decreases the marginal routing cost of that bin.

Objective:
    f(S) = sum_{i in S} revenue_i - alpha * 2 * sum_{i in S} dist(i, S \ {i})

We use a greedy algorithm with a max-priority queue to select bins effectively.
"""

import heapq
from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("supermodular_greedy")
class SupermodularGreedySelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Supermodular Greedy maximization (Synergy).
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Maximizes synergetic profit using greedy selection with re-evaluation.
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SupermodularGreedySelection"})

        # 1. Pre-compute individual revenues
        bin_cap = context.bin_volume * context.bin_density
        revenue = (context.current_fill / context.max_fill) * bin_cap * context.revenue_kg

        # 2. TSP lower bound components
        if context.distance_matrix is None:
            raise ValueError("SupermodularGreedySelection requires a distance_matrix.")

        alpha = context.modular_alpha
        budget = context.modular_budget if context.modular_budget > 0 else n_bins

        # 3. Greedy Selection
        # Initial set S = {0} (depot)
        selected_set: List[int] = []
        current_s = {0}  # indices into distance_matrix

        # Marginal gain g(i | S) = revenue_i - 2 * alpha * min_{j in S} dist(i, j)
        def get_marginal_gain(idx, S_indices):
            d_to_S = np.min(context.distance_matrix[idx + 1, list(S_indices)])
            return revenue[idx] - (alpha * 2.0 * d_to_S), SearchContext.initialize(
                selection_metrics={"strategy": "SupermodularGreedySelection"}
            )

        # Heap elements: (-gain, bin_index, version)
        priority_queue: List[Tuple[float, int, int]] = []
        for i in range(n_bins):
            # Initial evaluation vs depot.
            # (Note: we use the exact depot-to-bin distance for initialization).
            g = revenue[i] - (alpha * 2.0 * context.distance_matrix[i + 1, 0])
            heapq.heappush(priority_queue, (-g, i, 0))

        # Track progress to avoid infinite cycles in supermodular maximization
        stalled_count = 0
        max_stalled = len(priority_queue)

        while priority_queue and len(selected_set) < budget:
            neg_gain, bin_idx, version = heapq.heappop(priority_queue)
            gain = -neg_gain

            # Re-evaluate logic for supermodular (gains INCREASE as S grows)
            if version < len(selected_set):
                gain = get_marginal_gain(bin_idx, current_s)
                heapq.heappush(priority_queue, (-gain, bin_idx, len(selected_set)))
                stalled_count = 0  # re-evaluation counts as potential progress
                continue

            # Skip items whose current marginal gain is non-positive
            if gain <= 0:
                # Don't discard — push back with current version so it gets
                # re-evaluated after S grows (supermodular gains can turn positive).
                heapq.heappush(priority_queue, (-gain, bin_idx, len(selected_set)))
                stalled_count += 1
                if stalled_count >= max_stalled:
                    break  # cycled through everything without progress
                continue

            # Accept bin
            selected_set.append(bin_idx)
            current_s.add(bin_idx + 1)
            stalled_count = 0

        return sorted((np.array(selected_set) + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "SupermodularGreedySelection"}
        )
