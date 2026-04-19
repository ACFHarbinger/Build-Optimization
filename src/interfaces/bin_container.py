"""Container providing bin state access.

This module defines the IBinContainer protocol for duck typing with
objects that manage waste bin states in simulations and environments.

Example:
    >>> from logic.src.interfaces.bin_container import IBinContainer
    >>> def check_overflow(bins: IBinContainer, threshold: float) -> torch.Tensor:
    ...     return bins.fill_levels > threshold
"""

from typing import Any, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class IBinContainer(Protocol):
    """Container providing bin state access.

    This protocol defines the interface for objects that manage waste bin states
    in the simulation and environment. It unifies dict-based and object-based
    bin containers to eliminate getattr/isinstance chains.

    **Replaces patterns like**:
        - getattr(bins, "c", bins.get("c") if isinstance(bins, ITraversable) else None)
        - hasattr(bins, "fill_levels") or "fill_levels" in bins

    Attributes:
        fill_levels: Current fill levels (batch, n_bins)
        wastes: Waste usage (batch, n_bins)

    Example:
        >>> def check_overflow(bins: IBinContainer, threshold: float) -> torch.Tensor:
        ...     return bins.fill_levels > threshold
    """

    @property
    def fill_levels(self) -> torch.Tensor:
        """Current fill levels for all bins.

        Returns:
            Tensor of shape (batch, n_bins) with fill levels in [0, 1]
        """
        ...

    @property
    def wastes(self) -> torch.Tensor:
        """Waste usage for all bins.

        Returns:
            Tensor of shape (batch, n_bins) with waste values
        """
        ...

    def update_fill_levels(self, visited: torch.Tensor) -> None:
        """Update bin states after collection.

        Args:
            visited: Binary tensor (batch, n_bins) indicating collected bins
        """
        ...

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get attribute by key with optional default.

        Provides dict-like access for backward compatibility with code
        that uses bins.get("fill_levels").

        Args:
            key: Attribute name (e.g., "fill_levels", "wastes")
            default: Value to return if key not found

        Returns:
            Attribute value or default
        """
        ...
