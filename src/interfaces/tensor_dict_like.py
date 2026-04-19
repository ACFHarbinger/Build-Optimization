"""Structural type for TensorDict-compatible containers.

This module defines the ITensorDictLike protocol for duck typing with
dict-like objects that store tensors and have batch dimension information.

Example:
    >>> from logic.src.interfaces.tensor_dict_like import ITensorDictLike
    >>> if isinstance(state, ITensorDictLike):
    ...     batch_size = state.batch_size
"""

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class ITensorDictLike(Protocol):
    """Structural type for TensorDict-compatible containers.

    This protocol defines the interface for dict-like objects that store tensors
    and have batch dimension information. It unifies TensorDict, nested dicts,
    and custom state containers used throughout the RL pipeline.

    **Replaces patterns like**:
        - hasattr(obj, 'batch_size') and hasattr(obj, 'get')
        - isinstance(obj, dict) and 'batch_size' in obj

    Attributes:
        batch_size: Batch dimensions of stored tensors

    Example:
        >>> def process_state(state: ITensorDictLike) -> torch.Tensor:
        ...     batch_size = state.batch_size
        ...     loc = state.get('loc')
        ...     return loc
    """

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get tensor by key with optional default.

        Args:
            key: Tensor key name
            default: Value to return if key not found

        Returns:
            Tensor or default value
        """
        ...

    def set(self, key: str, value: torch.Tensor) -> None:
        """Set tensor by key.

        Args:
            key: Tensor key name
            value: Tensor to store
        """
        ...

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access tensor via indexing.

        Args:
            key: Tensor key name

        Returns:
            Tensor value

        Raises:
            KeyError: If key not found
        """
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Tensor key name

        Returns:
            True if key exists
        """
        ...

    def values(self) -> Any:
        """Return all tensor values.

        Returns:
            Iterable of tensor values
        """
        ...

    def keys(self) -> Any:
        """Return available tensor keys.

        Returns:
            Iterable of key names
        """
        ...

    def items(self) -> Any:
        """Return key-value pairs.

        Returns:
            Iterable of (key, value) tuples
        """
        ...

    @property
    def batch_size(self) -> Tuple[int, ...]:
        """Batch dimensions.

        Returns:
            Tuple of batch dimension sizes
        """
        ...

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored.

        Returns:
            Device (cpu, cuda, etc.)
        """
        ...
