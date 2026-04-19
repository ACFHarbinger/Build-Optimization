"""
IRouteImprovement — Route Improvement Interface.

Defines the contract for all route improvement / local-search operators.
Implementations MUST NOT hold mutable state that accumulates between
``process()`` calls; all intermediate variables belong in the returned
``ImprovementMetrics`` dict.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics


class IRouteImprovement(ABC):
    """
    Interface for all routing route improvers.

    All concrete implementations must:

    1. Not store algorithm state as instance attributes that mutate during
       ``process()``.  Configuration (time limits, seeds, etc.) may be
       stored at construction time.
    2. Return a ``Tuple[List[int], ImprovementMetrics]`` from ``process()``.
       The ``ImprovementMetrics`` dict is the mechanism for exporting
       telemetry to the ``SearchContext`` without coupling to it directly.
    """

    def __init__(self, **kwargs: Any):
        """Initialise route improver with configuration.

        Args:
            **kwargs: Configuration values.  If a single ``config`` key is
                passed, its value is used as the config dict directly.
        """
        if "config" in kwargs and len(kwargs) == 1:
            self.config = kwargs["config"]
        else:
            self.config = kwargs

    @abstractmethod
    def process(
        self,
        tour: List[int],
        **kwargs: Any,
    ) -> Tuple[List[int], ImprovementMetrics]:
        """
        Refine a given tour and return improvement telemetry.

        Args:
            tour: Initial tour (list of bin IDs including depot ``0``s).
            **kwargs: Context dictionary containing at minimum
                ``distance_matrix``.  May contain ``wastes``, ``capacity``,
                ``R``, ``C``, ``seed``, and other solver-specific keys.

        Returns:
            Tuple[List[int], ImprovementMetrics]:
                - ``refined_tour``: The best tour found.
                - ``metrics``: Telemetry for this improvement pass, suitable
                  for merging into ``SearchContext.improvement_metrics``.
        """
