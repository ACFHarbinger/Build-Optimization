"""
Context Package for Functional State Tracking.

Provides the typed ``SearchContext`` ledger and associated ``TypedDict``
metrics that flow immutably through the three-phase pipeline:

  Phase 1 — Mandatory Selection  : creates ``SearchContext``
  Phase 2 — Route Construction   : merges ``ConstructionMetrics`` / ``AcceptanceMetrics``
  Phase 3 — Route Improvement    : appends ``ImprovementMetrics``
"""

from logic.src.interfaces.context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    JointSelectionConstructionContext,
    MultiDayContext,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)

__all__ = [
    "SearchContext",
    "SearchPhase",
    "SelectionMetrics",
    "ConstructionMetrics",
    "AcceptanceMetrics",
    "ImprovementMetrics",
    "merge_context",
    "MultiDayContext",
    "JointSelectionConstructionContext",
]
