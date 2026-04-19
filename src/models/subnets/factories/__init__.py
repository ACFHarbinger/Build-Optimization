"""
Factory pattern for neural components (encoders, decoders).
"""

from .attention import AttentionComponentFactory
from .base import NeuralComponentFactory, _create_decoder_by_type
from .gac import GACComponentFactory
from .gcn import GCNComponentFactory
from .gfacs import GFACSComponentFactory
from .ggac import GGACComponentFactory
from .mdam import MDAMComponentFactory
from .mlp import MLPComponentFactory
from .moe import MoEComponentFactory
from .nar import NARComponentFactory
from .tgc import TGCComponentFactory

__all__ = [
    "NeuralComponentFactory",
    "AttentionComponentFactory",
    "GCNComponentFactory",
    "GACComponentFactory",
    "TGCComponentFactory",
    "GGACComponentFactory",
    "MLPComponentFactory",
    "MoEComponentFactory",
    "NARComponentFactory",
    "MDAMComponentFactory",
    "GFACSComponentFactory",
    "_create_decoder_by_type",
]
