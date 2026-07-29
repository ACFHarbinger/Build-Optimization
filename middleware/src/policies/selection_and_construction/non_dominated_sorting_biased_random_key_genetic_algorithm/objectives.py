"""
Multi-Objective Evaluator Module.

This module provides :func:`evaluate_chromosome` and
:func:`compute_overflow_risk`, the two primary objective-evaluation
utilities for the NDS-BRKGA optimiser.

Objective Space
---------------
The NDS-BRKGA minimises a **three-dimensional objective vector**:

1. ``f1 = −net_profit``  (→ maximise revenue − travel cost)
2. ``f2 = overflow_cost`` (→ minimise unselected bins' expected overflow loss)
3. ``f3 = total_distance`` (→ minimise total travel distance)

The first two objectives are in direct tension: selecting more bins increases
revenue (reducing ``f1``) but also reduces overflow cost (reducing ``f2``).
The NSGA-II Pareto-front extraction resolves this by finding the non-dominated
set, giving the solver a diverse portfolio of trade-off solutions.

Overflow Risk Computation
-------------------------
Per-bin overflow risk (used for both ``f2`` and adaptive threshold generation)
is computed identically to :func:`~logic.src.policies.mandatory_selection.selection_mip_knapsack._compute_overflow_risk`:

  ``score_i = E[overflow_kg_i] + P_i × penalty_frac × bin_mass_i``

where the expectation is taken over the ``ScenarioTree`` when available, and
over the current fill level as a deterministic proxy otherwise.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    Chromosome,
)

# ---------------------------------------------------------------------------
# Overflow risk computation
# ---------------------------------------------------------------------------


def compute_overflow_risk(
    current_fill: np.ndarray,
    bin_mass: np.ndarray,
    scenario_tree: Optional[Any],
    overflow_penalty_frac: float,
) -> np.ndarray:
    """
    Compute per-bin overflow risk scores.

    The score represents the *cost of NOT collecting* bin ``i`` (in kg
    of waste lost or penalised):

    .. math::

        \\text{score}_i = \\mathbb{E}[\\text{overflow\\_kg}_i]
                        + P_i \\times \\text{penalty\\_frac} \\times m_i

    where :math:`P_i` is the probability that at least one scenario
    overflows and :math:`m_i` is the bin's full capacity in kg.

    Args:
        current_fill: Current fill levels as percentages. Shape ``(N,)``.
        bin_mass: Per-bin full waste capacity in kg. Shape ``(N,)``.
        scenario_tree: Optional ``ScenarioTree`` from the prediction engine.
            If ``None``, current fill is used as a single deterministic
            scenario.
        overflow_penalty_frac: Penalty per overflow *event* expressed as a
            fraction of the bin's full capacity.

    Returns:
        np.ndarray: Per-bin overflow risk scores. Shape ``(N,)``.
            All values are non-negative.
    """
    n_bins = len(current_fill)

    if scenario_tree is None or not hasattr(scenario_tree, "get_scenarios_at_day"):
        # Deterministic fallback
        overflow_pct = np.maximum(0.0, current_fill - 100.0)
        overflow_kg = (overflow_pct / 100.0) * bin_mass
        overflow_prob = (current_fill >= 100.0).astype(float)
        occurrence_penalty = overflow_prob * overflow_penalty_frac * bin_mass
        return overflow_kg + occurrence_penalty

    expected_overflow_kg = np.zeros(n_bins, dtype=float)
    overflow_prob_any = np.zeros(n_bins, dtype=float)
    horizon: int = getattr(scenario_tree, "horizon", 1)

    for day in range(1, horizon + 1):
        try:
            day_nodes = scenario_tree.get_scenarios_at_day(day)
        except Exception:
            continue

        for node in day_nodes:
            prob: float = float(getattr(node, "probability", 0.0))
            if prob <= 0.0:
                continue
            wastes_arr = np.asarray(getattr(node, "wastes", np.empty(0)), dtype=float)
            if wastes_arr.size == 0:
                continue
            n_sc = min(n_bins, wastes_arr.size)
            overflow_pct = np.maximum(0.0, wastes_arr[:n_sc] - 100.0)
            overflow_kg_sc = (overflow_pct / 100.0) * bin_mass[:n_sc]
            expected_overflow_kg[:n_sc] += prob * overflow_kg_sc
            overflow_prob_any[:n_sc] = np.minimum(
                1.0,
                overflow_prob_any[:n_sc] + prob * (wastes_arr[:n_sc] >= 100.0).astype(float),
            )

    occurrence_penalty = overflow_prob_any * overflow_penalty_frac * bin_mass
    return expected_overflow_kg + occurrence_penalty


# ---------------------------------------------------------------------------
# Chromosome evaluation
# ---------------------------------------------------------------------------


def evaluate_chromosome(
    chrom: Chromosome,
    thresholds: np.ndarray,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    overflow_risk: np.ndarray,
    overflow_penalty: float,
    mandatory_override: Optional[List[int]] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate a chromosome and return its three-objective vector.

    Args:
        chrom: The chromosome to evaluate.
        thresholds: Per-bin adaptive selection thresholds (shape ``(N,)``).
        dist_matrix: Full ``(N+1)×(N+1)`` distance matrix (row/col 0 = depot).
        wastes: ``{1-based_bin_id: fill_pct}`` mapping for all bins.
        capacity: Vehicle capacity in the same units as waste values.
        R: Revenue per unit of fill collected.
        C: Cost per unit of distance.
        overflow_risk: Per-bin overflow risk scores from
            :func:`compute_overflow_risk`. Shape ``(N,)``.
        overflow_penalty: Scalar weight applied to the overflow objective.
        mandatory_override: 1-based bin IDs that must be visited regardless
            of the chromosome's selection keys.

    Returns:
        Tuple ``(neg_profit, overflow_cost, total_distance)`` where all
        three values are to be **minimised** by the NSGA-II sorter.
    """
    # --- Decode routes ---
    routes = chrom.to_routes(thresholds, wastes, capacity, mandatory_override)

    # Flat set of visited bins (1-based)
    visited: set = {node for route in routes for node in route}

    # --- Revenue ---
    revenue = sum(wastes.get(b, 0.0) * R for b in visited)

    # --- Travel cost + distance ---
    total_dist = 0.0
    dm = dist_matrix

    for route in routes:
        if not route:
            continue
        total_dist += dm[0, route[0]]
        for k in range(len(route) - 1):
            total_dist += dm[route[k], route[k + 1]]
        total_dist += dm[route[-1], 0]

    travel_cost = total_dist * C

    # --- Overflow cost for unselected bins ---
    # Only the bins that were NOT visited contribute to overflow cost.
    n_bins = chrom.n_bins
    unselected_mask = np.ones(n_bins, dtype=bool)
    for b in visited:
        if 1 <= b <= n_bins:
            unselected_mask[b - 1] = False
    overflow_cost = float(overflow_risk[unselected_mask].sum()) * overflow_penalty

    # --- Objectives (all to minimise) ---
    net_profit = revenue - travel_cost
    neg_profit = -net_profit

    return float(neg_profit), float(overflow_cost), float(total_dist)


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------


def evaluate_population(
    population: List[Chromosome],
    thresholds: np.ndarray,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    overflow_risk: np.ndarray,
    overflow_penalty: float,
    mandatory_override: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Evaluate all chromosomes in *population* and return the objective matrix.

    Args:
        population: List of ``Chromosome`` objects.
        thresholds: Per-bin adaptive thresholds. Shape ``(N,)``.
        dist_matrix: Full distance matrix.
        wastes: Fill-percentage mapping.
        capacity: Vehicle capacity.
        R: Revenue per fill unit.
        C: Cost per distance unit.
        overflow_risk: Per-bin overflow risk. Shape ``(N,)``.
        overflow_penalty: Weight for overflow objective.
        mandatory_override: Bins that must always be visited.

    Returns:
        np.ndarray: Objective matrix of shape ``(P, 3)`` where ``P`` is
            ``len(population)``.  Columns are ``[neg_profit, overflow_cost,
            total_distance]``, all to be minimised.
    """
    objectives = np.empty((len(population), 3), dtype=float)
    for idx, chrom in enumerate(population):
        objectives[idx] = evaluate_chromosome(
            chrom,
            thresholds,
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            overflow_risk,
            overflow_penalty,
            mandatory_override,
        )
    return objectives
