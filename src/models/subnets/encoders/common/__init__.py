"""
Common shared components for encoder implementations.

This package contains base classes and utilities shared across multiple encoder
implementations to reduce code duplication.
"""

from .encoder_base import TransformerEncoderBase
from .feed_forward_sublayer import EncoderFeedForwardSubLayer
from .multi_head_attention_layer import MultiHeadAttentionLayerBase

__all__ = [
    "TransformerEncoderBase",
    "EncoderFeedForwardSubLayer",
    "MultiHeadAttentionLayerBase",
]
