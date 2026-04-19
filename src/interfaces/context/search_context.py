"""
SearchContext — Functional State Tracking Ledger.

This module defines the canonical, typed ledger object (``SearchContext``) that
flows immutably through the three-phase heuristic pipeline.  No algorithm class
may hold mutable state that belongs in this ledger (e.g. ``self.info = {}``).

Architecture
------------
Phase 1 — Mandatory Selection
    Creates a new ``SearchContext`` via ``SearchContext.initialize()``,
    populating ``selection_metrics`` with intermediate outputs
    (attention tensors, knapsack matrices, bin importance scores).

Phase 2 — Route Construction
    Receives the ``SearchContext`` from Phase 1.  Merges its own
    ``ConstructionMetrics`` and, on each acceptance-criterion call,
    the returned ``AcceptanceMetrics`` via ``merge_context()``.

Phase 3 — Route Improvement
    Receives the ``SearchContext`` from Phase 2.  Appends
    ``ImprovementMetrics`` (trajectory statistics, local-optima counts).
    As the terminal node in the pipeline, the final context may be
    logged to a tracker (e.g., Weights & Biases) or discarded.

Immutability Contract
---------------------
``SearchContext`` is a frozen dataclass.  Modifications MUST go through
``merge_context(ctx, **patch)`` which returns a **new** instance.
This prevents downstream mutation errors during local-search rollbacks.

Tensor Safety
-------------
Any ``numpy.ndarray`` or ``torch.Tensor`` stored in a metrics dict MUST be
CPU-resident and detached from the autograd graph before insertion:

    tensor.detach().cpu()

This prevents memory leaks in the computation graph over thousands of
simulation iterations.

Example
-------
>>> from logic.src.interfaces.context import SearchContext, SelectionMetrics, merge_context
>>> sm: SelectionMetrics = {"bin_scores": [0.9, 0.3, 0.7], "strategy": "fractional_knapsack"}
>>> ctx = SearchContext.initialize(selection_metrics=sm)
>>> ctx2 = merge_context(ctx, construction_metrics={"insertion_costs": [1.2, 0.8]})
"""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Phase Enum
# ---------------------------------------------------------------------------


class SearchPhase(str, Enum):
    """Marks which pipeline phase last wrote to the ``SearchContext``."""

    SELECTION = "selection"
    CONSTRUCTION = "construction"
    IMPROVEMENT = "improvement"


# ---------------------------------------------------------------------------
# Metric TypedDicts — use plain Dict[str, Any] for Python 3.9 compat.
# The docstrings below define the *expected* schema for each phase.
# ---------------------------------------------------------------------------

# SelectionMetrics — populated during Mandatory Selection (Phase 1).
# Expected keys (all optional depending on strategy):
#   "strategy"          str         — name of the selection strategy used
#   "bin_scores"        List[float] — per-bin priority / score vector
#   "attention_map"     ndarray     — ML attention tensor (detached, CPU)
#   "knapsack_matrix"   ndarray     — intermediate assignment matrix
#   "n_selected"        int         — number of bins selected
#   "selection_time_s"  float       — wall-clock time for this phase
SelectionMetrics = Dict[str, Any]

# ConstructionMetrics — populated during Route Construction (Phase 2).
# Expected keys (all optional):
#   "algorithm"         str         — name of the construction algorithm
#   "insertion_costs"   List[float] — per-step cheapest-insertion delta
#   "feasibility_snap"  ndarray     — feasibility matrix at convergence (detached, CPU)
#   "n_iterations"      int         — iterations the constructive loop ran
#   "construction_time_s" float     — wall-clock time for this phase
ConstructionMetrics = Dict[str, Any]

# AcceptanceMetrics — populated on each IAcceptanceCriterion.accept() call.
# Expected keys (all optional, criterion-specific):
#   "criterion"         str         — registered name of the criterion (e.g. "bmc")
#   "temperature"       float       — current temperature (SA-style criteria)
#   "water_level"       float       — current water level (GD-style criteria)
#   "threshold"         float       — current threshold (RRT, TA, etc.)
#   "accepted"          bool        — outcome of this specific accept() call
#   "delta"             float       — candidate_obj - current_obj
#   "step_count"        int         — number of criterion steps taken so far
AcceptanceMetrics = Dict[str, Any]

# ImprovementMetrics — populated during Route Improvement (Phase 3).
# Expected keys (all optional):
#   "algorithm"         str         — name of the improvement algorithm
#   "n_iterations"      int         — total iterations executed
#   "n_local_optima"    int         — times the search hit a local optimum
#   "best_delta"        float       — total improvement over initial tour cost
#   "acceptance_trace"  List[AcceptanceMetrics] — per-step criterion telemetry
#   "improvement_time_s" float      — wall-clock time for this phase
ImprovementMetrics = Dict[str, Any]


# ---------------------------------------------------------------------------
# SearchContext — the immutable ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchContext:
    """
    Immutable state ledger that flows through the three-phase pipeline.

    All fields are frozen at creation time.  Use ``merge_context()`` to
    produce a new instance with updated fields.

    Attributes:
        phase: The pipeline phase that last wrote to this context.
        selection_metrics: Intermediate outputs from Mandatory Selection.
        construction_metrics: Intermediate outputs from Route Construction.
        acceptance_trace: Ordered list of per-step ``AcceptanceMetrics``
            emitted by ``IAcceptanceCriterion.accept()`` during construction
            or improvement.
        improvement_metrics: Statistics from Route Improvement.
        metadata: Catch-all for run-level annotations (e.g. day index, seed).
    """

    phase: SearchPhase = SearchPhase.SELECTION
    selection_metrics: SelectionMetrics = field(default_factory=dict)
    construction_metrics: ConstructionMetrics = field(default_factory=dict)
    acceptance_trace: List[AcceptanceMetrics] = field(default_factory=list)
    improvement_metrics: ImprovementMetrics = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        selection_metrics: Optional[SelectionMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SearchContext":
        """
        Factory: create a fresh ``SearchContext`` at the start of Phase 1.

        Args:
            selection_metrics: Outputs from the mandatory-selection strategy.
            metadata: Run-level annotations (e.g. ``{"day": 3, "seed": 42}``).

        Returns:
            A new ``SearchContext`` in the ``SELECTION`` phase.
        """
        return cls(
            phase=SearchPhase.SELECTION,
            selection_metrics=selection_metrics or {},
            construction_metrics={},
            acceptance_trace=[],
            improvement_metrics={},
            metadata=metadata or {},
        )

    def merge(self, other: "SearchContext") -> "SearchContext":
        """Merge this context with another by combining all metrics and metadata."""
        return merge_context(
            self,
            phase=other.phase,
            selection_metrics=other.selection_metrics,
            construction_metrics=other.construction_metrics,
            improvement_metrics=other.improvement_metrics,
            metadata=other.metadata,
        )


# TODO: Concatenate acceptance_trace if needed in a future update.


# ---------------------------------------------------------------------------
# Functional merge helper
# ---------------------------------------------------------------------------


def merge_context(
    ctx: SearchContext,
    phase: Optional[SearchPhase] = None,
    selection_metrics: Optional[SelectionMetrics] = None,
    construction_metrics: Optional[ConstructionMetrics] = None,
    acceptance_metrics: Optional[AcceptanceMetrics] = None,
    improvement_metrics: Optional[ImprovementMetrics] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SearchContext:
    """
    Return a **new** ``SearchContext`` with the specified fields updated.

    This is the only sanctioned mutation path.  All existing fields are
    shallow-copied; the caller's original ``ctx`` is never modified.

    For ``acceptance_metrics``  a single step's metrics dict is **appended**
    to the running ``acceptance_trace`` list, rather than replacing it.

    For ``Dict``-typed metrics fields (``selection_metrics``,
    ``construction_metrics``, ``improvement_metrics``) the patch is
    *merged* (shallow ``{**existing, **patch}``) rather than replaced.

    Args:
        ctx: The current (immutable) context snapshot.
        phase: New phase to set.  Defaults to the existing phase.
        selection_metrics: Patch to merge into ``ctx.selection_metrics``.
        construction_metrics: Patch to merge into ``ctx.construction_metrics``.
        acceptance_metrics: One step's acceptance telemetry to append.
        improvement_metrics: Patch to merge into ``ctx.improvement_metrics``.
        metadata: Patch to merge into ``ctx.metadata``.

    Returns:
        A new ``SearchContext`` with the requested updates applied.
    """
    new_selection = {**ctx.selection_metrics, **(selection_metrics or {})}
    new_construction = {**ctx.construction_metrics, **(construction_metrics or {})}
    new_improvement = {**ctx.improvement_metrics, **(improvement_metrics or {})}
    new_metadata = {**ctx.metadata, **(metadata or {})}

    # Append the new acceptance step without mutating the frozen list
    new_trace: List[AcceptanceMetrics]
    if acceptance_metrics is not None:
        new_trace = list(ctx.acceptance_trace) + [copy.copy(acceptance_metrics)]
    else:
        new_trace = list(ctx.acceptance_trace)

    return SearchContext(
        phase=phase if phase is not None else ctx.phase,
        selection_metrics=new_selection,
        construction_metrics=new_construction,
        acceptance_trace=new_trace,
        improvement_metrics=new_improvement,
        metadata=new_metadata,
    )
