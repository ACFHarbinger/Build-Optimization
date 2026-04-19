"""
MIP Multiple-Knapsack Selection Strategy Module — Overflow-Minimising Variant.

Exact 0/1 multiple-knapsack formulation of mandatory bin selection solved with
``scipy.optimize.milp``.  The objective is **minimisation** of expected overflow
loss across a lookahead horizon supplied as a ``ScenarioTree``.

Formulation
-----------
Let :math:`S` be the set of scenarios in the tree (leaves or nodes at all
future days), each with probability :math:`\\pi_s`.  For every bin *i* and
scenario *s*, let :math:`w_i^{(s)}` be the projected fill level *after* the
current day if bin *i* is **not** collected.  The overflow for bin *i* in
scenario *s* is:

.. math::

    o_i^{(s)} = \\max(0,\\; w_i^{(s)} - 100)

expressed as a fraction of full bin capacity (kg).

We select a binary vector :math:`x_i \\in \\{0,1\\}` (1 = collect today) that
**minimises**:

.. math::

    \\sum_{i} (1 - x_i) \\Bigl[
        \\underbrace{\\sum_s \\pi_s \\cdot o_i^{(s)} \\cdot \\hat m_i}_{\\text{expected waste lost}}
        +
        \\underbrace{P_i \\cdot \\Pr[\\text{any overflow}_i]}_{\\text{overflow occurrence penalty}}
    \\Bigr]

subject to the vehicle capacity and assignment constraints.

Because minimising :math:`(1-x_i) \\cdot r_i` is equivalent to maximising
:math:`x_i \\cdot r_i`, the formulation passes ``c_obj = -risk_i`` to ``milp``.

Overflow occurrence penalty
---------------------------
The parameter ``overflow_penalty_frac`` (default ``1.0``) scales the penalty
as a fraction of the bin's full waste capacity (kg).  A value of ``1.0`` means
one full bin-load of waste is penalised *per overflow event* in addition to the
expected spilled waste.

Fallback (no ScenarioTree)
--------------------------
When ``context.scenario_tree`` is ``None`` the strategy falls back to the
*current-fill* overflow proxy: the expected overflow is approximated as
:math:`\\max(0, u_i - 100) \\times \\hat m_i` and ``Pr[overflow]`` is 1 if
:math:`u_i \\ge 100` else 0, where :math:`u_i` is the current fill percentage.

Symmetry breaking
-----------------
Lexicographic count constraints ``sum_i x_{i,k} >= sum_i x_{i,k+1}`` prune
symmetric vehicle assignments.

Unbounded case
--------------
If ``context.n_vehicles <= 0`` the strategy short-circuits to all bins deemed
risky (non-zero expected overflow score).

Example::

    >>> from logic.src.policies.mandatory_selection.selection_mip_knapsack import (
    ...     MIPKnapsackSelection,
    ... )
    >>> strategy = MIPKnapsackSelection()
    >>> bins, ctx = strategy.select_bins(context)
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import (
    MandatorySelectionRegistry,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_overflow_risk(
    current_fill: np.ndarray,
    bin_mass: np.ndarray,
    scenario_tree: Optional[Any],
    overflow_penalty_frac: float,
) -> np.ndarray:
    """Compute per-bin overflow risk score (lower = safer to skip collecting).

    The score represents the *cost of NOT collecting* bin *i* today:

    .. code-block::

        score_i = E[overflow_kg_i] + P_i * overflow_penalty_frac * bin_mass_i

    where:
    - ``E[overflow_kg_i]`` = probability-weighted spilled waste in kg across
      all scenario-tree paths.
    - ``P_i`` = probability that *at least one* scenario overflows.
    - ``overflow_penalty_frac * bin_mass_i`` = occurrence penalty in kg.

    Args:
        current_fill: Current fill levels as percentages for each bin.
        bin_mass: Full capacity in kg for each bin.
        scenario_tree: Optional ``ScenarioTree`` (prediction module).  When
            ``None`` the current fill is used as the single deterministic scenario.
        overflow_penalty_frac: Penalty expressed as fraction of bin capacity added
            per overflow probability unit.

    Returns:
        np.ndarray: Per-bin overflow risk score (shape ``(n_bins,)``).
    """
    n_bins = len(current_fill)

    if scenario_tree is None or not hasattr(scenario_tree, "get_scenarios_at_day"):
        # Deterministic fallback: treat current fill as the only scenario.
        overflow_pct = np.maximum(0.0, current_fill - 100.0)  # percentage points
        overflow_kg = (overflow_pct / 100.0) * bin_mass
        overflow_prob = (current_fill >= 100.0).astype(float)
        occurrence_penalty = overflow_prob * overflow_penalty_frac * bin_mass
        return overflow_kg + occurrence_penalty

    # -----------------------------------------------------------------------
    # Aggregate over all future-day scenarios in the tree.
    # -----------------------------------------------------------------------
    # We collect every node at day >= 1 (not the root day=0, which is the
    # current state that will be set by visiting or not visiting today).
    # ``ScenarioTreeNode.wastes`` is a numpy array of fill levels [0-100].
    expected_overflow_kg = np.zeros(n_bins, dtype=float)
    overflow_prob_any = np.zeros(n_bins, dtype=float)

    # Determine horizon (number of future days in the tree).
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

            wastes = np.asarray(getattr(node, "wastes", np.empty(0)), dtype=float)
            if wastes.size == 0:
                continue

            # Align shapes: the scenario tree may contain a subset of bins.
            n_sc = min(n_bins, wastes.size)
            overflow_pct = np.maximum(0.0, wastes[:n_sc] - 100.0)
            overflow_kg_scenario = (overflow_pct / 100.0) * bin_mass[:n_sc]

            expected_overflow_kg[:n_sc] += prob * overflow_kg_scenario
            # Soft "at least once" probability accumulation (bounded at 1).
            overflow_prob_any[:n_sc] = np.minimum(
                1.0,
                overflow_prob_any[:n_sc] + prob * (wastes[:n_sc] >= 100.0).astype(float),
            )

    occurrence_penalty = overflow_prob_any * overflow_penalty_frac * bin_mass
    return expected_overflow_kg + occurrence_penalty


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@MandatorySelectionRegistry.register("mip_knapsack")
class MIPKnapsackSelection(IMandatorySelectionStrategy):
    """Exact 0/1 multiple-knapsack selection minimising expected overflow loss.

    Args:
        overflow_penalty_frac: Additional penalty per overflow event expressed
            as a fraction of the bin's full waste capacity.  Default ``1.0``
            (one full bin-load of waste).
    """

    def __init__(self, overflow_penalty_frac: float = 1.0) -> None:
        self.overflow_penalty_frac = overflow_penalty_frac

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """Solve the overflow-minimising 0/1 multiple-knapsack MIP.

        Reads ``vehicle_capacity``, ``n_vehicles``, ``bin_volume``,
        ``bin_density``, ``max_fill``, and (optionally) ``scenario_tree``
        from the context.

        Args:
            context: ``SelectionContext`` with bin-level inputs.

        Returns:
            Tuple[List[int], SearchContext]:
                - 1-based bin IDs selected for collection.
                - ``SearchContext`` carrying ``SelectionMetrics``.

        Raises:
            ValueError: If ``distance_matrix`` is not supplied.
        """
        if context.distance_matrix is None:
            raise ValueError("MIPKnapsackSelection requires a distance_matrix.")

        # Per-bin physical capacity (kg).
        bin_cap = context.bin_volume * context.bin_density  # kg per bin

        # Current fill in kg (used for capacity constraints).
        mass_all: np.ndarray = (context.current_fill / context.max_fill) * bin_cap

        # Overflow risk score for every bin (cost of NOT collecting).
        risk_all: np.ndarray = _compute_overflow_risk(
            current_fill=context.current_fill,
            bin_mass=np.full_like(mass_all, bin_cap),
            scenario_tree=context.scenario_tree,
            overflow_penalty_frac=context.overflow_penalty_frac,
        )

        # Only consider bins with a non-zero overflow risk and positive mass.
        eligible_idx = np.nonzero((mass_all > 0) & (risk_all > 0))[0]
        if eligible_idx.size == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "MIPKnapsackSelection"})

        n_vehicles = int(getattr(context, "n_vehicles", 1))

        # Unbounded knapsacks: take all risky bins.
        if n_vehicles <= 0:
            return sorted((eligible_idx + 1).tolist()), SearchContext.initialize(
                selection_metrics={"strategy": "MIPKnapsackSelection"}
            )

        capacity = float(context.vehicle_capacity)
        if capacity <= 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "MIPKnapsackSelection"})

        # Dominance prune: drop bins that do not fit any single vehicle.
        fits_any = mass_all[eligible_idx] <= capacity
        eligible_idx = eligible_idx[fits_any]
        if eligible_idx.size == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "MIPKnapsackSelection"})

        n = int(eligible_idx.size)
        K = n_vehicles  # K = number of vehicles

        # Objective coefficients: maximise risk reduction <=> minimise -(risk).
        # milp minimises c @ x, so c = -risk (we want to select high-risk bins).
        r = risk_all[eligible_idx]  # per-bin overflow risk scores
        m = mass_all[eligible_idx]  # per-bin masses (for capacity constraints)

        # Decision variables flattened row-major: idx(i, k) = i * K + k.
        n_vars = n * K

        # Objective: minimise -sum_{i,k} r_i * x_{i,k}  (== maximise total risk reduction).
        c_obj = -np.repeat(r, K)

        # Assignment: each bin is collected by at most one vehicle.
        A_assign = np.kron(np.eye(n), np.ones((1, K)))

        # Capacity: total mass in vehicle k must not exceed Q.
        A_cap = np.kron(m[None, :], np.eye(K))

        constraints = [
            LinearConstraint(A_assign, -np.inf, 1.0),
            LinearConstraint(A_cap, -np.inf, capacity),
        ]

        # Lexicographic symmetry break: count(k) >= count(k+1).
        if K >= 2:
            A_sym = np.zeros((K - 1, n_vars))
            for k in range(K - 1):
                for i in range(n):
                    A_sym[k, i * K + k] = 1.0
                    A_sym[k, i * K + k + 1] = -1.0
            constraints.append(LinearConstraint(A_sym, 0.0, np.inf))

        integrality = np.ones(n_vars, dtype=int)  # all binary
        bounds = Bounds(lb=0, ub=1)

        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if not result.success or result.x is None:
            return [], SearchContext.initialize(selection_metrics={"strategy": "MIPKnapsackSelection"})

        x = np.asarray(result.x).reshape(n, K)
        taken_local = np.nonzero(x.sum(axis=1) > 0.5)[0]
        taken_global = eligible_idx[taken_local]
        return sorted((taken_global + 1).tolist()), SearchContext.initialize(
            selection_metrics={
                "strategy": "MIPKnapsackSelection",
                "n_selected": int(taken_local.size),
                "total_risk_reduced": float(r[taken_local].sum()),
            }
        )
