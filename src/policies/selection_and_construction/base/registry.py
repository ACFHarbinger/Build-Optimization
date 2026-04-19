"""
Joint Policy Registry Module.

Provides a string-keyed registry for
:class:`~logic.src.policies.selection_and_construction.base.base_joint_policy.BaseJointPolicy`
subclasses, enabling plugin-style extension of joint selection-and-construction
solvers.

Example::

    >>> from logic.src.policies.selection_and_construction.base.joint_registry import (
    ...     JointPolicyRegistry,
    ... )
    >>> @JointPolicyRegistry.register("my_joint")
    ... class MyJointPolicy(BaseJointPolicy): ...
    >>> cls = JointPolicyRegistry.get("my_joint")
"""

from typing import Dict, List, Optional, Type


class JointPolicyRegistry:
    """Registry for joint selection-and-construction policy classes.

    Attributes:
        _registry: Internal mapping from lower-case name to policy class.
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers *policy_cls* under *name*.

        Args:
            name: Unique string identifier for the policy.

        Returns:
            The class decorator.
        """

        def decorator(policy_cls: Type) -> Type:
            cls._registry[name.lower()] = policy_cls
            return policy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Retrieve a policy class by name.

        Args:
            name: The string key used during registration.

        Returns:
            The policy class, or ``None`` if not found.
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_policies(cls) -> List[str]:
        """Return the names of all registered joint policies.

        Returns:
            List of registered name strings.
        """
        return list(cls._registry.keys())
