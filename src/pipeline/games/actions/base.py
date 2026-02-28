"""
Abstract base class for game optimization actions (Command Pattern).

Mirrors SimulationAction in the simulation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class GameAction(ABC):
    """
    Abstract base for game optimization commands.

    Each action receives the shared context dict and modifies it in-place,
    following the same Command Pattern used in SimulationAction.
    """

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the action and update the context in-place.

        Args:
            context: Shared dictionary containing optimization state.
        """
