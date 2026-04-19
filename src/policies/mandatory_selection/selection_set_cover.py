"""
Set-Cover Selection Strategy Module.

Treats the selection problem as a Minimum Set Cover instance. The goal is to
select a minimum number of "hub" bins such that every bin exceeding the
critical threshold is within a specified service radius of at least one hub.
This is implemented using a greedy heuristic which provides a ln(n)
approximation ratio.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_set_cover import SetCoverSelection
    >>> strategy = SetCoverSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("set_cover")
class SetCoverSelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on greedy Set Cover.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select hub bins to cover all critical bins within service_radius.

        Args:
            context: SelectionContext with critical_threshold and service_radius.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if context.distance_matrix is None:
            raise ValueError("SetCoverSelection requires a distance_matrix.")

        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SetCoverSelection"})

        # 1. Define the Universe U of critical bins
        fill_ratios = context.current_fill / context.max_fill
        critical_indices = np.nonzero(fill_ratios > context.critical_threshold)[0]

        if len(critical_indices) == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SetCoverSelection"})

        universe = set(critical_indices)

        # 2. Build candidate sets for each bin j
        # cover(j) = { i in U : distance(i, j) <= service_radius }
        # matrix is (n_bins+1, n_bins+1), depot at 0.
        # dist_matrix_bins is (n_bins, n_bins)
        dist_matrix_bins = context.distance_matrix[1:, 1:]

        candidate_sets = []
        for j in range(n_bins):
            # Find indices i in critical_indices that are within radius of j
            distances_from_j = dist_matrix_bins[j, critical_indices]
            covered_in_universe = critical_indices[np.nonzero(distances_from_j <= context.service_radius)[0]]
            candidate_sets.append(set(covered_in_universe))

        # 3. Greedy Selection
        selected_hubs = []
        uncovered = universe.copy()

        while uncovered:
            best_j = -1
            best_count = -1

            # Find candidate j that covers most still-uncovered elements
            for j in range(n_bins):
                count = len(candidate_sets[j].intersection(uncovered))
                if count > best_count:
                    best_count = count
                    best_j = j

            if best_j == -1 or best_count == 0:
                # Should not happen as long as critical_threshold < 1 and
                # radius >= 0 (a bin covers itself)
                break

            selected_hubs.append(best_j)
            uncovered = uncovered - candidate_sets[best_j]

        return sorted((np.array(selected_hubs) + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "SetCoverSelection"}
        )
