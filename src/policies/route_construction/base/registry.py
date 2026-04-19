"""
Route Constructor Registry Module.

This module provides a central registry for all route constructor implementations,
allowing for dynamic discovery and instantiation of route constructors by string names.

Attributes:
    RouteConstructorRegistry: The singleton registry class.

Example:
    >>> from logic.src.policies.route_construction.base import RouteConstructorRegistry
    >>> @RouteConstructorRegistry.register("my_route_constructor")
    ... class MyPolicy(IRouteConstructor): ...
    >>> policy_cls = RouteConstructorRegistry.get("my_policy")
"""

from typing import Callable, Dict, List, Optional, Type

from logic.src.interfaces.route_constructor import IRouteConstructor


# --- Route Constructor Registry ---
class RouteConstructorRegistry:
    """
    Central registry for route constructors.

    Attributes:
        _registry (Dict[str, Type[IRouteConstructor]]): Internal dictionary mapping policy names to classes.
    """

    _registry: Dict[str, Type[IRouteConstructor]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a policy class.

        Args:
            name: Unique identifier for the policy.

        Returns:
            Callable: The decorator function.
        """

        def decorator(policy_cls: Type[IRouteConstructor]) -> Type[IRouteConstructor]:
            """
            Inner decorator function.

            Args:
                policy_cls: The policy implementation class.

            Returns:
                Type[IRouteConstructor]: The registered policy class.
            """
            cls._registry[name] = policy_cls
            return policy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[IRouteConstructor]]:
        """
        Retrieve a policy class by name.

        Args:
            name: The name of the policy to retrieve.

        Returns:
            Optional[Type[IRouteConstructor]]: The policy class if found, else None.
        """
        return cls._registry.get(name)

    @classmethod
    def list_route_constructors(cls) -> List[str]:
        """
        List all registered route constructors.

        Returns:
            List[str]: A list of names of all registered route constructors.
        """
        return list(cls._registry.keys())
