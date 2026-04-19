"""
Common shared components for decoder implementations.

This package contains base classes and utilities shared across multiple decoder
implementations to reduce code duplication.
"""

from .cache import AttentionDecoderCache
from .feed_forward_sublayer import FeedForwardSubLayer
from .selection import select_action, select_action_log_prob

__all__ = [
    "AttentionDecoderCache",
    "FeedForwardSubLayer",
    "select_action",
    "select_action_log_prob",
]
