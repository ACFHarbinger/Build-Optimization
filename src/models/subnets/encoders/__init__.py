"""
Encoder Subnetworks.
"""

from .common import (
    EncoderFeedForwardSubLayer,
    MultiHeadAttentionLayerBase,
    TransformerEncoderBase,
)
from .deepaco import DeepACOEncoder
from .gac import GraphAttConvEncoder
from .gat import GraphAttentionEncoder
from .gcn import GraphConvolutionEncoder
from .gfacs import GFACSEncoder
from .ggac import GatedGraphAttConvEncoder
from .matnet import MatNetEncoder
from .mdam import MDAMGraphAttentionEncoder
from .mlp import MLPEncoder
from .moe import MoEGraphAttentionEncoder
from .nargnn import NARGNNEncoder, NARGNNNodeEncoder
from .ptr import PointerEncoder
from .tgc import TransGraphConvEncoder

__all__ = [
    # Base classes and common components
    "TransformerEncoderBase",
    "EncoderFeedForwardSubLayer",
    "MultiHeadAttentionLayerBase",
    # Encoder implementations
    "DeepACOEncoder",
    "GraphAttConvEncoder",
    "GraphAttentionEncoder",
    "GraphConvolutionEncoder",
    "GFACSEncoder",
    "GatedGraphAttConvEncoder",
    "MatNetEncoder",
    "MDAMGraphAttentionEncoder",
    "MLPEncoder",
    "MoEGraphAttentionEncoder",
    "NARGNNEncoder",
    "NARGNNNodeEncoder",
    "PointerEncoder",
    "TransGraphConvEncoder",
]
