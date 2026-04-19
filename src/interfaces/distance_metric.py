from abc import ABC, abstractmethod
from typing import Any


class IDistanceMetric(ABC):
    """
    Abstract Base Class for structural distance metrics between optimization solutions.
    Used for topography and diversity aware exploration (e.g., Skewed VNS).
    """

    @abstractmethod
    def compute(self, current: Any, candidate: Any) -> float:
        """
        Computes the structural distance between two solutions.

        Args:
            current (Any): The current solution representation (e.g., list of routes).
            candidate (Any): The candidate solution representation.

        Returns:
            float: The scalar structural distance.
        """
        pass
