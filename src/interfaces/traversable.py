"""Config-like objects supporting nested traversal.

This module defines the ITraversable protocol for duck typing with
configuration objects (dict, DictConfig, dataclass).

Example:
    >>> from logic.src.interfaces.traversable import ITraversable
    >>> def extract_param(config: ITraversable, key: str) -> Any:
    ...     if key in config:
    ...         return config[key]
    ...     return config.get(key, None)
"""

from typing import Any, Iterator, Optional, Protocol, TypeVar, Union, overload, runtime_checkable

_T = TypeVar("_T")


@runtime_checkable
class ITraversable(Protocol):
    """Config-like objects supporting nested traversal.

    This protocol unifies dict, DictConfig (OmegaConf), and dataclass objects
    that need to be traversed in configuration parsing. It eliminates repeated
    isinstance checks for dict/DictConfig/list patterns.

    **Replaces patterns like**:
        - isinstance(config, dict) or isinstance(config, DictConfig)
        - hasattr(config, 'keys') and hasattr(config, 'get')

    Example:
        >>> def extract_param(config: ITraversable, key: str) -> Any:
        ...     if hasattr(config, "get"):
        ...         return config.get(key, None)
        ...     return None
    """

    def __getitem__(self, key: Any, /) -> Any:
        """Get value by key. Positional-only to match dict."""
        ...

    def __contains__(self, key: Any, /) -> bool:
        """Check if key exists. Positional-only to match dict."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Support iteration over keys (required for mapping-like behavior)."""
        ...

    def __len__(self) -> int:
        """Return the number of items."""
        ...

    def keys(self) -> Any:
        """Return available configuration keys.

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

    def values(self) -> Any:
        """Return all values.

        Returns:
            Iterable of values
        """
        ...

    @overload
    def get(self, key: str, /) -> Optional[Any]: ...

    @overload
    def get(self, key: str, default: _T, /) -> Union[Any, _T]: ...

    def get(self, key: str, default: Any = None, /) -> Any:
        """Get value by key with optional default.

        Args:
            key: Configuration key (string)
            default: Value to return if key not found

        Returns:
            Configuration value or default
        """
        ...
