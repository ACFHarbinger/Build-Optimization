"""
Edge embedding modules for graph-based encoders.
"""

from torch import nn

from .base import EdgeEmbedding
from .cvrpp import CVRPPEdgeEmbedding
from .none import NoEdgeEmbedding
from .tsp import TSPEdgeEmbedding
from .wcvrp import WCVRPEdgeEmbedding

EDGE_EMBEDDING_REGISTRY = {
    "vrpp": TSPEdgeEmbedding,
    "cvrpp": CVRPPEdgeEmbedding,
    "wcvrp": WCVRPEdgeEmbedding,
    "cwcvrp": WCVRPEdgeEmbedding,
    "sdwcvrp": WCVRPEdgeEmbedding,
    "swcvrp": WCVRPEdgeEmbedding,
    "scwcvrp": WCVRPEdgeEmbedding,
    "none": NoEdgeEmbedding,
}


def get_edge_embedding(
    env_name: str,
    embed_dim: int = 128,
    **kwargs,
) -> nn.Module:
    """
    Get problem-specific edge embedding module.

    Args:
        env_name: Environment/problem name.
        embed_dim: Edge embedding dimension.
        **kwargs: Additional arguments passed to the embedding constructor.

    Returns:
        Initialized edge embedding module.
    """
    if env_name not in EDGE_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown environment for edge embedding: {env_name}. Available: {list(EDGE_EMBEDDING_REGISTRY.keys())}"
        )
    return EDGE_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim, **kwargs)


__all__ = [
    "EdgeEmbedding",
    "CVRPPEdgeEmbedding",
    "NoEdgeEmbedding",
    "TSPEdgeEmbedding",
    "WCVRPEdgeEmbedding",
    "EDGE_EMBEDDING_REGISTRY",
    "get_edge_embedding",
]
