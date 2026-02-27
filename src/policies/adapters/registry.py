"""
Policy Registry Module.

This module provides a central registry for all routing policy implementations,
allowing for dynamic discovery and instantiation of policies by string names.

Attributes:
    PolicyRegistry: The singleton registry class.

Example:
    >>> from logic.src.policies.adapters.registry import PolicyRegistry
    >>> @PolicyRegistry.register("my_policy")
    ... class MyPolicy(IPolicy): ...
    >>> policy_cls = PolicyRegistry.get("my_policy")
"""

from typing import Callable, Dict, List, Optional, Type

from logic.src.interfaces.policy import IPolicy


# --- Policy Registry ---
class PolicyRegistry:
    """
    Central registry for routing policies.

    Attributes:
        _registry (Dict[str, Type[IPolicy]]): Internal dictionary mapping policy names to classes.
    """

    _registry: Dict[str, Type[IPolicy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a policy class.

        Args:
            name: Unique identifier for the policy.

        Returns:
            Callable: The decorator function.
        """

        def decorator(policy_cls: Type[IPolicy]) -> Type[IPolicy]:
            """
            Inner decorator function.

            Args:
                policy_cls: The policy implementation class.

            Returns:
                Type[IPolicy]: The registered policy class.
            """
            cls._registry[name] = policy_cls
            return policy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[IPolicy]]:
        """
        Retrieve a policy class by name.

        Args:
            name: The name of the policy to retrieve.

        Returns:
            Optional[Type[IPolicy]]: The policy class if found, else None.
        """
        return cls._registry.get(name)

    @classmethod
    def list_policies(cls) -> List[str]:
        """
        List all registered policies.

        Returns:
            List[str]: A list of names of all registered policies.
        """
        return list(cls._registry.keys())
