"""
Interfaces for Build Optimization components.
DEFINING PROTOCOLS TO DECOUPLE MODULES.
"""

from .adapter import IPolicyAdapter
from .env import IEnv
from .model import IModel
from .policy import IPolicy
from .tensor_dict_like import ITensorDictLike
from .traversable import ITraversable

__all__ = [
    "IPolicyAdapter",
    "IEnv",
    "IModel",
    "IPolicy",
    "ITensorDictLike",
    "ITraversable",
]
