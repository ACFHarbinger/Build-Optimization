"""
Mandatory Base Package.

This package defines the core infrastructure for the "Mandatory" selection
policy, including the context object, factory pattern for strategy creation,
and the registry for available strategies.

Attributes:
    MandatorySelectionFactory (class): Factory for creating strategies.
    MandatorySelectionRegistry (class): Registry for strategy classes.
    SelectionContext (class): Data container for selection context.

Example:
    >>> from logic.src.policies.mandatory.base import SelectionContext
    >>> ctx = SelectionContext(bin_ids=..., current_fill=...)
"""

from .selection_context import SelectionContext
from .selection_factory import MandatorySelectionFactory
from .selection_registry import MandatorySelectionRegistry

__all__ = ["MandatorySelectionFactory", "MandatorySelectionRegistry", "SelectionContext"]
