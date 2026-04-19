"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

from .model import PolyNet as PolyNet
from .policy import PolyNetPolicy as PolyNetPolicy

__all__ = ["PolyNet", "PolyNetPolicy"]
