"""
Selection Registry Module.

This module provides a registry for `IMandatorySelectionStrategy` classes.
It allows dynamic registration and retrieval of strategies, facilitating
plugin-style extensions.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.base.selection_registry import MandatorySelectionRegistry
    >>> @MandatorySelectionRegistry.register("my_strategy")
    >>> class MyStrategy(IMandatorySelectionStrategy): ...
"""

from typing import Dict, Optional, Type

from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy


class MandatorySelectionRegistry:
    """Registry for Mandatory selection strategies."""

    _strategies: Dict[str, Type[IMandatorySelectionStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy."""

        def wrapper(strategy_cls: Type[IMandatorySelectionStrategy]):
            """Register the class with the given name."""
            cls._strategies[name.lower()] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[IMandatorySelectionStrategy]]:
        """Get a strategy class by name."""
        return cls._strategies.get(name.lower())

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategy names."""
        return list(cls._strategies.keys())
