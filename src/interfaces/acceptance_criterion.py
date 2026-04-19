"""
IAcceptanceCriterion — Move Acceptance Interface.

Defines the contract for all pluggable move-acceptance strategies used
during route construction and improvement.  Every concrete implementation
MUST be stateless between ``accept()`` calls; internal state (temperature,
water level, etc.) is advanced only via ``step()``.

The ``accept()`` method now returns a ``Tuple[bool, AcceptanceMetrics]``
so that callers can thread per-step telemetry into the ``SearchContext``
without polling ``get_state()`` separately.  ``get_state()`` is retained for
backward compatibility but is superseded by the metrics in the returned tuple.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Tuple, Union

from logic.src.interfaces.context.search_context import AcceptanceMetrics

# Type alias for single or multi-objective values
ObjectiveValue = Union[float, Sequence[float]]


class IAcceptanceCriterion(ABC):
    """
    Abstract Base Class for modular move acceptance strategies.

    Implementations MUST NOT store mutable state that changes during
    ``accept()`` itself.  State advances exclusively via ``step()``.
    """

    @abstractmethod
    def setup(self, initial_objective: ObjectiveValue) -> None:
        """
        Initialise the internal state of the criterion from the starting solution.

        Args:
            initial_objective (ObjectiveValue): The objective value of the starting solution.
        """

    @abstractmethod
    def accept(
        self,
        current_obj: ObjectiveValue,
        candidate_obj: ObjectiveValue,
        **kwargs: Any,
    ) -> Tuple[bool, AcceptanceMetrics]:
        """
        Evaluate whether a candidate solution should be accepted.

        This method MUST be free of side-effects.  All state that changes
        between steps is advanced exclusively in ``step()``.

        Args:
            current_obj (ObjectiveValue): The objective value of the current incumbent.
            candidate_obj (ObjectiveValue): The objective value of the proposed candidate.
            **kwargs (Any): Additional context (e.g. structural representations).

        Returns:
            Tuple[bool, AcceptanceMetrics]:
                - ``is_accepted``: True if the candidate is accepted.
                - ``metrics``: Telemetry snapshot for this decision step,
                  suitable for appending to ``SearchContext.acceptance_trace``.
        """

    @abstractmethod
    def step(
        self,
        current_obj: ObjectiveValue,
        candidate_obj: ObjectiveValue,
        accepted: bool,
        **kwargs: Any,
    ) -> None:
        """
        Advance the internal state of the criterion (e.g. cool temperature,
        update memory buffers, adjust thresholds).

        Args:
            current_obj (ObjectiveValue): The objective after the decision was applied.
            candidate_obj (ObjectiveValue): The objective of the evaluated candidate.
            accepted (bool): The boolean result of ``accept()``.
            **kwargs (Any): Additional context.
        """

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve internal variables for telemetry.

        .. deprecated::
            Superseded by the ``AcceptanceMetrics`` returned from ``accept()``.
            Retained for backward compatibility only.

        Returns:
            Dict[str, Any]: Dictionary of internal state variables.
        """
        return {}
