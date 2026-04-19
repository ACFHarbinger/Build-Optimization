"""
Filter-and-Fan Selection Strategy Module.

Implements the Filter-and-Fan metaheuristic (Glover, 1998) adapted for mandatory
bin selection. The algorithm operates in two phases:

  Filter phase  – scores every bin by a composite urgency/profit criterion and
                  retains only the top-k candidates (the filter beam).
  Fan phase     – from the filtered seed set, explores a structured add/remove
                  neighbourhood for up to `ff_fan_depth` passes, greedily
                  accepting any move that improves net-profit. The iteration
                  terminates early when no improving move is found.

References:
    Glover, F. (1998). A template for scatter search and path relinking.
    Lecture Notes in Computer Science, 1415, 1-51.
"""

from typing import List, Set, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("filter_and_fan")
class FilterAndFanSelection(IMandatorySelectionStrategy):
    """
    Bin selection strategy based on the Filter-and-Fan metaheuristic.

    Filter phase: rank bins by a composite urgency-profit score and keep the
    top-k (controlled by ``context.ff_filter_width``) as the seed solution.

    Fan phase: refine the seed solution by greedily evaluating add and remove
    moves for up to ``context.ff_fan_depth`` passes. The separable net-profit
    objective makes each move evaluation O(1), so the full fan phase runs in
    O(n * ff_fan_depth) time.

    Parameters (read from SelectionContext):
        ff_filter_width: Number of top-scoring bins kept after the filter phase.
                         Defaults to max(5, n_bins // 3).
        ff_fan_depth:    Maximum number of add/remove sweep passes in the fan
                         phase. Defaults to 3.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins using the Filter-and-Fan procedure.

        Args:
            context: SelectionContext with fill levels, revenue, and cost data.

        Returns:
            Tuple of selected bin IDs (1-based) and a SearchContext with metrics.
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "FilterAndFanSelection"})

        filter_width: int = context.ff_filter_width if context.ff_filter_width > 0 else max(5, n_bins // 3)
        fan_depth: int = context.ff_fan_depth

        # ── Per-bin economics ────────────────────────────────────────────────
        bin_cap = context.bin_volume * context.bin_density
        revenues = (context.current_fill / context.max_fill) * bin_cap * context.revenue_kg

        if context.distance_matrix is not None:
            # Row 0 = depot; columns 1..n correspond to bins 0..n-1
            round_trip_cost = 2.0 * context.distance_matrix[0, 1:] * context.cost_per_km
        else:
            round_trip_cost = np.zeros(n_bins)

        # Separable net-profit: positive means the bin contributes value
        net_profit = revenues - round_trip_cost

        # ── FILTER PHASE ─────────────────────────────────────────────────────
        # Composite score: net profit amplified by fill urgency so bins that
        # are both profitable AND nearly full are ranked higher.
        urgency = context.current_fill / context.max_fill  # in [0, 1]
        scores = net_profit * (1.0 + urgency)

        k = min(filter_width, n_bins)
        top_k_idx = np.argsort(scores)[::-1][:k]
        best_set: Set[int] = set(top_k_idx.tolist())

        # ── FAN PHASE ────────────────────────────────────────────────────────
        # Objective is separable: O(1) per add/remove move.
        def objective(S: Set[int]) -> float:
            return float(np.sum(net_profit[list(S)])) if S else 0.0

        best_obj = objective(best_set)
        all_idx = set(range(n_bins))

        for _ in range(fan_depth):
            improved = False

            # Try adding each bin not yet selected
            for j in sorted(all_idx - best_set, key=lambda x: -net_profit[x]):
                delta = net_profit[j]
                if delta > 0 and best_obj + delta > best_obj:
                    best_set.add(j)
                    best_obj += delta
                    improved = True

            # Try removing each selected bin
            for j in sorted(list(best_set), key=lambda x: net_profit[x]):
                delta = net_profit[j]
                if delta <= 0:
                    best_set.discard(j)
                    best_obj -= delta
                    improved = True

            if not improved:
                break

        # Always collect bins that are at or above the urgency threshold
        if context.threshold > 0.0:
            forced = set(np.nonzero(context.current_fill >= context.threshold)[0].tolist())
            best_set |= forced

        selected = sorted((np.array(sorted(best_set), dtype=np.int32) + 1).tolist())

        return selected, SearchContext.initialize(
            selection_metrics={
                "strategy": "FilterAndFanSelection",
                "filter_width": k,
                "fan_depth": fan_depth,
                "objective": round(best_obj, 4),
                "n_selected": len(selected),
            }
        )
