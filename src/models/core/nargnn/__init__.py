"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

from .model import NARGNN as NARGNN
from .policy import NARGNNPolicy as NARGNNPolicy

__all__ = ["NARGNN", "NARGNNPolicy"]
