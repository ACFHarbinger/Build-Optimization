"""
BRKGA Chromosome Module.

Defines :class:`Chromosome`, the fundamental unit of the NDS-BRKGA population.

Encoding
--------
A chromosome is a vector ``keys`` of length ``2 * N`` where ``N`` is the number
of candidate bins:

* **Selection keys** ``keys[0:N]``:  bin ``i`` (0-indexed) is selected if
  ``keys[i] > threshold_i``, where ``threshold_i`` is computed **per-bin**
  from the normalised overflow risk (adaptive threshold).

* **Routing keys** ``keys[N:2N]``:  the visit order of the selected bins is
  determined by ``argsort(keys[N + j] for j in selected_indices)`` — lower
  routing key → visited earlier.

Adaptive Threshold
------------------
The per-bin selection threshold is::

    threshold_i = threshold_max - (threshold_max - threshold_min)
                  × normalised_risk_i

where ``normalised_risk_i = overflow_risk_i / max(overflow_risk)`` is in
``[0, 1]``.  A bin with maximum overflow risk therefore has
``threshold_i = threshold_min`` (≈ 0.10), making any random key above 0.10
trigger its selection.  A bin with zero risk has ``threshold_i = threshold_max``
(≈ 0.90), requiring a very high key before being selected.

Route Decoding
--------------
After selection, a greedy capacity-constrained packing phase splits the ordered
sequence of selected bins into feasible vehicle routes.  The packing respects
``capacity`` in kg (using the fill percentages converted to mass via
``bin_density × bin_volume / 100``).

References:
    Gonçalves, J. F., & Resende, M. G. (2011).
        Biased random-key genetic algorithms for combinatorial optimization.
        *Journal of Heuristics*, 17(5), 487–525.
"""

from typing import Dict, List, Optional

import numpy as np


def compute_adaptive_thresholds(
    overflow_risk: np.ndarray,
    threshold_min: float = 0.10,
    threshold_max: float = 0.90,
) -> np.ndarray:
    """
    Compute per-bin adaptive selection thresholds from overflow risk scores.

    The threshold is inversely proportional to the normalised overflow risk:
    a high-risk bin gets a low threshold (easy to select); a zero-risk bin
    gets a high threshold (hard to select).

    If all bins have identical risk (e.g., all zero) the thresholds default
    to ``threshold_max`` (no bin is preferentially selected).

    Args:
        overflow_risk: Per-bin overflow risk scores.  Shape ``(N,)``.
            Values must be non-negative; scale is arbitrary (normalised internally).
        threshold_min: Threshold for the bin with maximum overflow risk.
        threshold_max: Threshold for bins with zero overflow risk.

    Returns:
        np.ndarray: Per-bin thresholds in ``[threshold_min, threshold_max]``.
            Shape ``(N,)``.
    """
    risk = np.asarray(overflow_risk, dtype=float)
    max_risk = risk.max()
    if max_risk <= 1e-12:
        # All bins equally safe: use upper threshold uniformly
        return np.full_like(risk, threshold_max)

    normalised = risk / max_risk  # in [0, 1]
    thresholds = threshold_max - (threshold_max - threshold_min) * normalised
    return thresholds  # shape (N,)


class Chromosome:
    """
    BRKGA chromosome encoding joint bin-selection and route-construction.

    Each chromosome stores ``2 * N`` continuous random keys in ``[0, 1]``.

    Attributes:
        keys: Float array of shape ``(2 * N,)``.  First half = selection
            keys; second half = routing keys.
        n_bins: Number of candidate bins ``N``.
    """

    __slots__ = ("keys", "n_bins")

    def __init__(self, keys: np.ndarray, n_bins: int) -> None:
        """
        Create a Chromosome from a pre-built key vector.

        Args:
            keys: Float array of length ``2 * n_bins``, all in ``[0, 1]``.
            n_bins: Number of candidate bins.

        Raises:
            ValueError: If ``len(keys) != 2 * n_bins``.
        """
        if len(keys) != 2 * n_bins:
            raise ValueError(f"keys length {len(keys)} != 2 * n_bins {2 * n_bins}")
        self.keys: np.ndarray = np.asarray(keys, dtype=float)
        self.n_bins: int = n_bins

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, n_bins: int, rng: np.random.Generator) -> "Chromosome":
        """
        Create a chromosome with uniformly random keys.

        Args:
            n_bins: Number of candidate bins.
            rng: NumPy random Generator instance.

        Returns:
            A new random Chromosome.
        """
        return cls(rng.uniform(0.0, 1.0, size=2 * n_bins), n_bins)

    @classmethod
    def from_selection_and_order(
        cls,
        n_bins: int,
        selected_bins_0idx: List[int],
        routing_order_0idx: List[int],
        overhead: float = 0.15,
        rng: Optional[np.random.Generator] = None,
    ) -> "Chromosome":
        """
        Encode a known solution as a chromosome for population seeding.

        Selected bins receive selection keys in ``[0.6 + overhead, 1.0]``
        (above threshold), unselected bins receive keys in
        ``[0.0, 0.4 - overhead]`` (below threshold).  Routing keys are
        assigned based on position in *routing_order_0idx* — earlier
        positions get smaller keys.

        Args:
            n_bins: Total number of candidate bins.
            selected_bins_0idx: 0-based indices of bins to select.
            routing_order_0idx: 0-based indices of selected bins, in the
                order they should be visited.
            overhead: Additional margin above/below the threshold midpoint
                to ensure robustness against minor key perturbations.
            rng: Optional RNG for jitter within the assigned bands.

        Returns:
            A seeded Chromosome encoding the given solution.
        """
        if rng is None:
            rng = np.random.default_rng()

        keys = np.zeros(2 * n_bins, dtype=float)

        selected_set = set(selected_bins_0idx)

        # Selection keys
        for i in range(n_bins):
            lo = 0.6 + overhead if i in selected_set else 0.0
            hi = 1.0 if i in selected_set else 0.4 - overhead
            keys[i] = rng.uniform(lo, hi)

        # Routing keys: map position in order → small key value [0, 0.5)
        n_sel = len(routing_order_0idx)
        for rank, b_idx in enumerate(routing_order_0idx):
            keys[n_bins + b_idx] = (rank + 1) / (n_sel + 1) * 0.5

        # Remaining (unselected) bins get random routing keys in (0.5, 1.0]
        for i in range(n_bins):
            if i not in selected_set:
                keys[n_bins + i] = rng.uniform(0.5, 1.0)

        return cls(keys, n_bins)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_selection(self, thresholds: np.ndarray) -> List[int]:
        """
        Decode the selection sub-vector using per-bin adaptive thresholds.

        Bin ``i`` (0-indexed) is selected if ``keys[i] > thresholds[i]``.

        Args:
            thresholds: Per-bin adaptive thresholds.  Shape ``(N,)``.
                Computed from overflow risk via :func:`compute_adaptive_thresholds`.

        Returns:
            Sorted list of **1-based** bin IDs selected for collection.
        """
        mask = self.keys[: self.n_bins] > thresholds
        return sorted((np.nonzero(mask)[0] + 1).tolist())

    def decode_routing_order(self, selected_1based: List[int]) -> List[int]:
        """
        Determine the visit sequence of selected bins from their routing keys.

        Lower routing key → earlier in the tour.

        Args:
            selected_1based: List of 1-based bin IDs chosen by the selection phase.

        Returns:
            The same IDs reordered by ascending routing key.
        """
        if not selected_1based:
            return []

        sel_0 = [b - 1 for b in selected_1based]
        routing_subkeys = self.keys[self.n_bins :][sel_0]
        order = np.argsort(routing_subkeys)
        return [selected_1based[i] for i in order]

    def to_routes(
        self,
        thresholds: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        mandatory_override: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Full decode: selection → routing order → capacity-constrained packing.

        The method packs the ordered sequence of selected bins into vehicle
        routes greedily, starting a new route whenever adding the next bin
        would exceed ``capacity``.  Bins in ``mandatory_override`` are
        guaranteed to be included regardless of their selection key.

        Args:
            thresholds: Per-bin adaptive thresholds from
                :func:`compute_adaptive_thresholds`.
            wastes: ``{1-based_bin_id: fill_pct}`` for all bins.
            capacity: Per-vehicle carrying capacity (same unit as waste
                values, typically kg).
            mandatory_override: 1-based bin IDs that MUST be visited.

        Returns:
            List of routes.  Each route is a list of **1-based** bin IDs
            (local sub-problem indices are **not** used here; global IDs
            are used instead so the caller can map them directly).  Routes
            do not include the depot sentinel (0).
        """
        # --- Selection phase ---
        selected = set(self.decode_selection(thresholds))
        if mandatory_override:
            selected.update(mandatory_override)
        selected_list: List[int] = sorted(selected)

        if not selected_list:
            return []

        # --- Routing phase: impose visit order from routing keys ---
        ordered = self.decode_routing_order(selected_list)

        # --- Packing phase: greedy capacity-constrained split ---
        routes: List[List[int]] = []
        current_route: List[int] = []
        current_load = 0.0

        for bin_id in ordered:
            load = wastes.get(bin_id, 0.0)
            if current_route and current_load + load > capacity + 1e-9:
                routes.append(current_route)
                current_route = [bin_id]
                current_load = load
            else:
                current_route.append(bin_id)
                current_load += load

        if current_route:
            routes.append(current_route)

        return routes
