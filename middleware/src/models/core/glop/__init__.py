"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

from .model import GLOP as GLOP
from .policy import GLOPPolicy as GLOPPolicy

__all__ = ["GLOP", "GLOPPolicy"]
