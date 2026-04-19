"""
Route Improvement Registry Module.

This module provides a registry for route improvement operators. It allows
registering and retrieving route improvers by name.

Attributes:
    RouteImproverRegistry (class): The registry class.

Example:
    >>> from logic.src.policies.helpers.route_improvement.base.registry import RouteImproverRegistry
    >>> RouteImproverRegistry.register("my_processor", MyProcessorClass)
    >>> cls = RouteImproverRegistry.get_route_improver_class("my_processor")
"""

from typing import Callable, Dict, Optional, Type

from logic.src.interfaces.route_improvement import IRouteImprovement


class RouteImproverRegistry:
    """Registry for routing route improvement strategies."""

    _strategies: Dict[str, Type[IRouteImprovement]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a route improver."""

        def wrapper(processor_cls: Type[IRouteImprovement]):
            """Wrapper for registering the processor class."""
            cls._strategies[name.lower()] = processor_cls
            return processor_cls

        return wrapper

    @classmethod
    def get_route_improver_class(cls, name: str) -> Optional[Type[IRouteImprovement]]:
        """Retrieve a route improver by name."""
        return cls._strategies.get(name.lower())

    @classmethod
    def list_improvers(cls) -> list:
        """List all registered improver names."""
        return list(cls._strategies.keys())
