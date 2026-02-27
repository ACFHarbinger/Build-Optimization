"""
Old Bachelor Acceptance (OBA) solver for VRPP.

Exports:
    OBASolver: The main solver class.
    OBAParams: Configuration parameters dataclass.
"""

from .params import OBAParams
from .solver import OBASolver

__all__ = ["OBASolver", "OBAParams"]
