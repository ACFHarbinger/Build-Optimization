"""
Route Improvement Base Package.

This package defines the core infrastructure for the "Route Improvement"
policy, including the factory pattern for algorithm creation,
and the registry for available algorithms.

Attributes:
    RouteImproverFactory (class): Factory for creating algorithms.
    RouteImproverRegistry (class): Registry for algorithm classes.

Example:
    >>> from logic.src.policies.helpers.route_improvement.base import RouteImproverFactory
    >>> factory = RouteImproverFactory()
    >>> algorithm = factory.create("fast_tsp")
"""

from logic.src.interfaces import IRouteImprovement

from .factory import RouteImproverFactory
from .registry import RouteImproverRegistry

__all__ = [
    "IRouteImprovement",
    "RouteImproverRegistry",
    "RouteImproverFactory",
]
