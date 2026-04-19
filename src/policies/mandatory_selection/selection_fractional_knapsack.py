"""
Fractional Multiple-Knapsack Selection Strategy Module.

Greedy selection for the multiple-knapsack formulation of mandatory bin
selection. Each knapsack represents one vehicle; its capacity is the
vehicle's waste-mass capacity ``context.vehicle_capacity``. Each bin
contributes a weight equal to its current waste mass and a value equal
to its *net profit* — the expected revenue minus the marginal distance
cost of collecting it.

Objective
---------
    maximize    sum_{i,k} ( r_i - lambda * w_{i,k}^{dist} ) * x_{i,k}
    subject to  sum_k  x_{i,k}                  <= 1       for all i
                sum_i  m_i * x_{i,k}            <= Q       for all k
                0 <= x_{i,k} <= 1

where ``m_i`` is bin mass, ``Q`` is vehicle capacity, ``r_i`` is expected
revenue, ``w_{i,k}^{dist}`` is the cheapest-insertion distance of bin
``i`` into the bins already packed in knapsack ``k`` (depot round-trip
if ``k`` is empty), and ``lambda = context.cost_per_km`` converts
distance into revenue units.

Marginal insertion cost
-----------------------
For the first bin inserted in a knapsack, the distance is the depot
round-trip ``2 * d(depot, b)``. For every subsequent bin, it is the
cheapest-insertion delta against the bin already selected that is
spatially closest to the candidate:

    delta(b | S) = d(c, b) + d(b, depot) - d(c, depot),
    where c = argmin_{s in S} d(s, b).

Approximation guarantee
-----------------------
Pure density-greedy is not a constant-factor approximation for
multiple-knapsack. We recover a 1/2-approximation by returning the
better of the greedy solution and the best single bin that fits alone,
via the standard ``OPT <= V(greedy) + V(best_single_item)`` argument.

Unbounded case
--------------
If ``context.n_vehicles <= 0`` the problem has no binding capacity
constraint and the strategy short-circuits to every revenue-positive
bin with non-negative net profit.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_fractional_knapsack \\
    ...     import FractionalKnapsackSelection
    >>> strategy = FractionalKnapsackSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Set, Tuple

import numpy as np

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("fractional_knapsack")
class FractionalKnapsackSelection(IMandatorySelectionStrategy):
    """Greedy net-profit density selection under per-vehicle mass capacity."""

    _EPS = 1e-9

    def _insertion_distances(
        self,
        candidates: np.ndarray,
        packed: List[int],
        dm: np.ndarray,
        dist_depot: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized marginal insertion distance for each candidate bin.

        Indices are 0-based bin indices; the distance_matrix places the
        depot at index 0 so bin ``i`` corresponds to matrix index ``i+1``.
        """
        if not packed:
            return 2.0 * dist_depot[candidates]

        packed_arr = np.asarray(packed, dtype=int)
        sub = dm[np.ix_(candidates + 1, packed_arr + 1)]
        nearest_pos = np.argmin(sub, axis=1)
        d_ic = sub[np.arange(candidates.size), nearest_pos]
        d_c_depot = dist_depot[packed_arr[nearest_pos]]
        deltas = d_ic + dist_depot[candidates] - d_c_depot
        return np.maximum(deltas, self._EPS)

    def _pack_one_knapsack(
        self,
        eligible: Set[int],
        revenue: np.ndarray,
        mass: np.ndarray,
        dm: np.ndarray,
        dist_depot: np.ndarray,
        cost_per_km: float,
        capacity: float,
    ) -> List[int]:
        """Greedily fill a single vehicle, mutating ``eligible`` in place."""
        packed: List[int] = []
        remaining = capacity

        while eligible:
            cand = np.fromiter(eligible, dtype=int, count=len(eligible))

            # Drop items that can no longer fit mass-wise.
            fits = mass[cand] <= remaining
            if not np.any(fits):
                break
            cand = cand[fits]

            dist = self._insertion_distances(cand, packed, dm, dist_depot)
            net_profit = revenue[cand] - cost_per_km * dist

            # Items with non-positive net profit never help the objective.
            positive = net_profit > 0
            if not np.any(positive):
                break
            cand = cand[positive]
            net_profit = net_profit[positive]

            density = net_profit / mass[cand]
            best = int(np.argmax(density))
            chosen = int(cand[best])

            eligible.discard(chosen)
            packed.append(chosen)
            remaining -= float(mass[chosen])

        return packed

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Return the selected bin IDs (1-based) under the multi-vehicle plan.

        Reads ``vehicle_capacity``, ``n_vehicles``, and (optionally)
        ``cost_per_km`` from the context. A non-positive ``n_vehicles``
        means the number of knapsacks is unbounded.
        """
        if context.distance_matrix is None:
            raise ValueError("FractionalKnapsackSelection requires a distance_matrix.")
        if context.revenue_kg <= 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "FractionalKnapsackSelection"})

        dm = np.asarray(context.distance_matrix)

        # Per-bin mass (kg) and revenue (currency).
        bin_cap = context.bin_volume * context.bin_density
        mass = (context.current_fill / context.max_fill) * bin_cap
        revenue = mass * context.revenue_kg
        dist_depot = np.asarray(dm[0, 1:], dtype=float)

        cost_per_km = float(getattr(context, "cost_per_km", 0.0))
        net_profit_depot = revenue - cost_per_km * (2.0 * dist_depot)

        # Eligible = positive mass and positive standalone net profit.
        eligible_mask = (mass > 0) & (net_profit_depot > 0)
        eligible_idx: Set[int] = set(int(i) for i in np.nonzero(eligible_mask)[0])
        if not eligible_idx:
            return [], SearchContext.initialize(selection_metrics={"strategy": "FractionalKnapsackSelection"})

        n_vehicles = int(getattr(context, "n_vehicles", 1))

        # Unbounded knapsacks: capacity does not bind.
        if n_vehicles <= 0:
            return sorted(i + 1 for i in eligible_idx), SearchContext.initialize(
                selection_metrics={"strategy": "FractionalKnapsackSelection"}
            )

        capacity = float(context.vehicle_capacity)
        if capacity <= 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "FractionalKnapsackSelection"})

        # Discard bins that individually exceed a single vehicle's capacity.
        eligible_idx = {i for i in eligible_idx if mass[i] <= capacity}
        if not eligible_idx:
            return [], SearchContext.initialize(selection_metrics={"strategy": "FractionalKnapsackSelection"})

        # --- Greedy pass across all K knapsacks ---
        remaining = set(eligible_idx)
        greedy_selected: List[int] = []
        for _ in range(n_vehicles):
            if not remaining:
                break
            greedy_selected.extend(
                self._pack_one_knapsack(remaining, revenue, mass, dm, dist_depot, cost_per_km, capacity)
            )
        greedy_value = float(np.sum(revenue[greedy_selected])) if greedy_selected else 0.0

        # --- Best-single-item fallback for the 1/2-approximation guarantee ---
        # OPT <= V(greedy) + V(best_single_item), so max(.,.) >= OPT / 2.
        elig_arr = np.fromiter(eligible_idx, dtype=int, count=len(eligible_idx))
        best_local = int(np.argmax(revenue[elig_arr]))
        best_single = int(elig_arr[best_local])
        best_single_value = float(revenue[best_single])

        if best_single_value > greedy_value:
            return [best_single + 1], SearchContext.initialize(
                selection_metrics={"strategy": "FractionalKnapsackSelection"}
            )
        return sorted(i + 1 for i in greedy_selected), SearchContext.initialize(
            selection_metrics={"strategy": "FractionalKnapsackSelection"}
        )
