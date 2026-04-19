"""
Problem-specific initialization embeddings.

Each problem type requires different input features to be embedded
before being processed by the encoder.
"""

from __future__ import annotations

from torch import nn

from logic.src.models.subnets.embeddings.cvrpp import CVRPPInitEmbedding
from logic.src.models.subnets.embeddings.vrpp import VRPPInitEmbedding
from logic.src.models.subnets.embeddings.wcvrp import WCVRPInitEmbedding

from .context import (
    CONTEXT_EMBEDDING_REGISTRY,
    ContextEmbedder,
    GenericContextEmbedder,
    VRPPContextEmbedder,
    WCVRPContextEmbedder,
)
from .dynamic import DynamicEmbedding
from .edges import (
    EDGE_EMBEDDING_REGISTRY,
    CVRPPEdgeEmbedding,
    EdgeEmbedding,
    NoEdgeEmbedding,
    TSPEdgeEmbedding,
    WCVRPEdgeEmbedding,
    get_edge_embedding,
)
from .positional import (
    POSITIONAL_EMBEDDING_REGISTRY,
    AbsolutePositionalEmbedding,
    CyclicPositionalEmbedding,
    pos_init_embedding,
)
from .state import (
    STATE_EMBEDDING_REGISTRY,
    CVRPPState,
    EnvState,
    SWCVRPState,
    VRPPState,
    WCVRPState,
)
from .static import StaticEmbedding

# Embedding registry
INIT_EMBEDDING_REGISTRY = {
    "vrpp": VRPPInitEmbedding,
    "cvrpp": CVRPPInitEmbedding,
    "wcvrp": WCVRPInitEmbedding,
    "cwcvrp": WCVRPInitEmbedding,
    "sdwcvrp": WCVRPInitEmbedding,
    "swcvrp": WCVRPInitEmbedding,
    "scwcvrp": WCVRPInitEmbedding,
}

DYNAMIC_EMBEDDING_REGISTRY = {
    "static": StaticEmbedding,
    "dynamic": DynamicEmbedding,
}


def get_init_embedding(env_name: str, embed_dim: int = 128) -> nn.Module:
    """
    Get problem-specific initial embedding layer.

    Args:
        env_name: Environment/problem name.
        embed_dim: Embedding dimension.

    Returns:
        Initialized embedding module.
    """
    if env_name not in INIT_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown environment for embedding: {env_name}. Available: {list(INIT_EMBEDDING_REGISTRY.keys())}"
        )
    return INIT_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim)


__all__ = [
    "VRPPInitEmbedding",
    "CVRPPInitEmbedding",
    "WCVRPInitEmbedding",
    "EnvState",
    "VRPPState",
    "CVRPPState",
    "WCVRPState",
    "SWCVRPState",
    "ContextEmbedder",
    "VRPPContextEmbedder",
    "WCVRPContextEmbedder",
    "DynamicEmbedding",
    "StaticEmbedding",
    "EdgeEmbedding",
    "TSPEdgeEmbedding",
    "CVRPPEdgeEmbedding",
    "WCVRPEdgeEmbedding",
    "NoEdgeEmbedding",
    "INIT_EMBEDDING_REGISTRY",
    "STATE_EMBEDDING_REGISTRY",
    "DYNAMIC_EMBEDDING_REGISTRY",
    "EDGE_EMBEDDING_REGISTRY",
    "get_init_embedding",
    "get_edge_embedding",
    "GenericContextEmbedder",
    "AbsolutePositionalEmbedding",
    "CyclicPositionalEmbedding",
    "pos_init_embedding",
    "POSITIONAL_EMBEDDING_REGISTRY",
    "CONTEXT_EMBEDDING_REGISTRY",
]
