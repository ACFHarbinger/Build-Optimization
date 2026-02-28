"""
Abstract base class for game optimization states.

Mirrors SimState in the simulation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import GameOptimizationContext


class GameState(ABC):
    """Abstract base class for game build optimization states."""

    context: "GameOptimizationContext"

    @abstractmethod
    def handle(self, ctx: "GameOptimizationContext") -> None:
        """Execute this state and transition to the next one."""
