"""
Economic Order Quantity (EOQ) threshold helper.

Computes the optimal fill-level trigger for each bin by balancing the
holding cost of waste already accumulated against the ordering cost of
dispatching a vehicle to collect it. Used by selection strategies that
want to derive their trigger threshold from cost parameters rather than
from a hand-tuned ``context.threshold`` value.

Classical EOQ (Harris, 1913) gives the economic batch size
    Q* = sqrt(2 * D * S / h)
where D is demand rate (kg/day), S is ordering cost per visit, and h is
holding cost per kg per day. We convert Q* to a fill-level ratio
    tau_i = min(1, Q* / bin_mass_capacity_i)
which is then returned in the same units as current_fill (typically percent).
"""

from __future__ import annotations

import numpy as np

from logic.src.policies.mandatory_selection.base.selection_context import (
    SelectionContext,
)


def compute_eoq_thresholds(context: SelectionContext) -> np.ndarray:
    """
    Return per-bin EOQ-derived fill-level triggers in absolute fill units.

    The returned triggers are scaled by context.max_fill to match the units
    of context.current_fill (typically percent 0-100).

    Falls back to a uniform threshold of ``context.threshold``
    when required parameters are missing or non-positive.
    """
    n = len(context.current_fill)

    h = float(getattr(context, "holding_cost_per_kg_day", 0.0))
    S = float(getattr(context, "ordering_cost_per_visit", 0.0))
    mu = context.accumulation_rates

    if h <= 0 or S <= 0 or mu is None:
        # Assumes context.threshold is in the same units as context.max_fill (percent)
        return np.full(n, float(context.threshold), dtype=float)

    bin_mass_cap = float(context.bin_volume) * float(context.bin_density)
    if bin_mass_cap <= 0:
        return np.full(n, float(context.threshold), dtype=float)

    # Convert mu (fill-percent/day) into kg/day using the same linear map
    # the rest of the codebase uses for mass.
    demand_kg_day = (np.asarray(mu) / float(context.max_fill)) * bin_mass_cap
    demand_kg_day = np.where(demand_kg_day <= 0, 1e-9, demand_kg_day)

    Q_star = np.sqrt(2.0 * demand_kg_day * S / h)

    # Ratio in [0, 1]
    tau_ratio = np.minimum(1.0, Q_star / bin_mass_cap)

    # Convert to absolute units (e.g. 0-100)
    tau = tau_ratio * float(context.max_fill)

    # Safety assertion
    assert np.all(tau <= context.max_fill + 1e-9), "EOQ threshold exceeds max_fill"

    return tau


def resolve_trigger_threshold(context: SelectionContext, fill_ratios: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of bins whose fill meets the effective trigger.

    If ``context.use_eoq_threshold`` is True, the per-bin EOQ threshold is
    used; otherwise a uniform ``context.threshold`` is applied.
    """
    # Important: context.current_fill is absolute (percent), fill_ratios is normalized.
    # We choose to compare in absolute units to match compute_eoq_thresholds output.
    current_fill = context.current_fill

    if getattr(context, "use_eoq_threshold", False):
        tau = compute_eoq_thresholds(context)
    else:
        tau = np.full_like(current_fill, float(context.threshold), dtype=float)

    return current_fill >= tau
