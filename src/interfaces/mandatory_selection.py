"""
IMandatorySelectionStrategy — Mandatory Selection Interface.

Defines the protocol for all strategies that determine which bins must
be collected on a given simulation day.  Each strategy creates a fresh
``SearchContext`` (Phase 1 of the pipeline) and returns it alongside
the list of selected bin IDs.
"""

from typing import TYPE_CHECKING, List, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from logic.src.interfaces.context.search_context import SearchContext
    from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext


@runtime_checkable
class IMandatorySelectionStrategy(Protocol):
    """Interface for Mandatory selection strategies.

    Implementors MUST return a ``SearchContext`` initialised with
    ``SelectionMetrics`` describing the intermediate computation.
    This context is passed unchanged into Phase 2 (Route Construction).
    """

    def select_bins(
        self,
        context: "SelectionContext",
    ) -> Tuple[List[int], "SearchContext"]:
        """
        Determine which bins must be collected and create a phase-1 ledger.

        Args:
            context: ``SelectionContext`` containing all bin-level inputs
                needed to evaluate the strategy.

        Returns:
            Tuple[List[int], SearchContext]:
                - ``selected_bins``: 1-based bin IDs to visit.
                - ``search_context``: Freshly initialised ``SearchContext``
                  carrying ``SelectionMetrics`` for downstream phases.
        """
        ...
