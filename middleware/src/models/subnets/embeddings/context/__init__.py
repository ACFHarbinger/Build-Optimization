"""
Unified context embedding modules for VRP variants.
"""

from .base import ContextEmbedder
from .generic import GenericContextEmbedder
from .vrpp import VRPPContextEmbedder
from .wcvrp import WCVRPContextEmbedder

CONTEXT_EMBEDDING_REGISTRY = {
    "vrpp": VRPPContextEmbedder,
    "cvrpp": VRPPContextEmbedder,
    "wcvrp": WCVRPContextEmbedder,
    "cwcvrp": WCVRPContextEmbedder,
    "sdwcvrp": WCVRPContextEmbedder,
    "swcvrp": WCVRPContextEmbedder,
    "scwcvrp": WCVRPContextEmbedder,
}

__all__ = [
    "ContextEmbedder",
    "GenericContextEmbedder",
    "VRPPContextEmbedder",
    "WCVRPContextEmbedder",
]
