"""
Unified cache for attention-based decoder architectures.

This module provides a unified cache structure that consolidates multiple
decoder-specific cache implementations:
- glimpse/fixed.py: AttentionModelFixed (5 fields)
- gat/deep_gat_cache.py: DeepAttentionModelFixed (2 fields)
- polynet/cache.py: PrecomputedCache (5 fields)
- mdam/cache.py: PrecomputedCache (5 fields)

The unified cache supports efficient slicing and is compatible with all
attention-based decoder types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class AttentionDecoderCache:
    """
    Unified cache for attention-based decoder precomputations.

    This cache stores precomputed projections of encoder outputs that remain
    fixed during autoregressive decoding. Using a cache avoids recomputing
    these projections at each decoding step, significantly improving performance.

    The cache supports different decoder architectures:
    - Full multi-head attention decoders (Glimpse, PolyNet, MDAM): Use all fields
    - Simplified decoders (DeepGAT): Use only node_embeddings + graph_context

    Parameters
    ----------
    node_embeddings : torch.Tensor
        Original node embeddings from encoder.
        Shape: (batch_size, num_nodes, embed_dim)
    graph_context : torch.Tensor
        Projected graph-level context (mean/aggregate of nodes).
        Shape: (batch_size, 1, embed_dim) or (batch_size, embed_dim)
        Also called "context_node_projected" in some decoders.
    glimpse_key : Optional[torch.Tensor], default=None
        Precomputed keys for multi-head glimpse attention.
        Shape: (batch_size, num_heads, num_nodes, key_dim)
        Used by: GlimpseDecoder, PolyNetDecoder, MDAMDecoder
    glimpse_val : Optional[torch.Tensor], default=None
        Precomputed values for multi-head glimpse attention.
        Shape: (batch_size, num_heads, num_nodes, val_dim)
        Used by: GlimpseDecoder, PolyNetDecoder, MDAMDecoder
    logit_key : Optional[torch.Tensor], default=None
        Precomputed keys for final logit computation.
        Shape: (batch_size, num_nodes, embed_dim) or (batch_size, num_heads, num_nodes, key_dim)
        Used by: GlimpseDecoder, PolyNetDecoder, MDAMDecoder

    Attributes
    ----------
    Same as parameters.

    Methods
    -------
    __getitem__(key)
        Slice the cache to select specific batch elements.

    Examples
    --------
    Create cache for full multi-head attention decoder:
    >>> cache = AttentionDecoderCache(
    ...     node_embeddings=node_emb,          # (B, N, D)
    ...     graph_context=graph_ctx,           # (B, 1, D)
    ...     glimpse_key=g_key,                 # (B, H, N, K)
    ...     glimpse_val=g_val,                 # (B, H, N, K)
    ...     logit_key=l_key,                   # (B, N, D)
    ... )

    Create cache for simplified decoder (DeepGAT):
    >>> cache = AttentionDecoderCache(
    ...     node_embeddings=node_emb,          # (B, N, D)
    ...     graph_context=graph_ctx,           # (B, 1, D)
    ... )

    Slice cache to get subset of batch:
    >>> sub_cache = cache[0:10]  # First 10 batch elements
    >>> single_cache = cache[5]  # 5th batch element

    Notes
    -----
    - Optional fields (glimpse_key, glimpse_val, logit_key) are set to None
      for decoders that don't use multi-head attention mechanisms.
    - The `graph_context` field is equivalent to `context_node_projected`
      in AttentionModelFixed from the glimpse decoder.
    - Slicing with an integer index returns a cache with batch_size=1
      (adds unsqueeze(0) to maintain batch dimension).
    """

    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: Optional[torch.Tensor] = None
    glimpse_val: Optional[torch.Tensor] = None
    logit_key: Optional[torch.Tensor] = None

    def __getitem__(self, key: Union[int, slice, torch.Tensor]) -> "AttentionDecoderCache":
        """
        Slice the cache to select specific batch elements.

        This method allows efficient batching and sub-batching during decoding.
        All tensor fields are sliced along the batch dimension (dimension 0).

        Parameters
        ----------
        key : int, slice, or torch.Tensor
            Index, slice, or boolean/integer tensor for selecting batch elements.
            - int: Select single batch element (adds unsqueeze(0))
            - slice: Select range of batch elements (e.g., [0:10])
            - torch.Tensor: Advanced indexing with boolean or integer tensor

        Returns
        -------
        AttentionDecoderCache
            New cache instance with sliced tensors.

        Examples
        --------
        >>> cache = AttentionDecoderCache(...)  # (B=32, N=50, D=128)
        >>> sub_cache = cache[0:16]             # (B=16, N=50, D=128)
        >>> single = cache[5]                   # (B=1, N=50, D=128) - unsqueezed
        >>> filtered = cache[mask]              # (B=?, N=50, D=128) - boolean indexing
        """
        # Handle integer indexing (add batch dimension back)
        if isinstance(key, int):
            return AttentionDecoderCache(
                node_embeddings=self.node_embeddings[key].unsqueeze(0),
                graph_context=self.graph_context[key].unsqueeze(0),
                glimpse_key=self.glimpse_key[key].unsqueeze(0) if self.glimpse_key is not None else None,
                glimpse_val=self.glimpse_val[key].unsqueeze(0) if self.glimpse_val is not None else None,
                logit_key=self.logit_key[key].unsqueeze(0) if self.logit_key is not None else None,
            )

        # Handle slice or tensor indexing (preserves batch dimension)
        return AttentionDecoderCache(
            node_embeddings=self.node_embeddings[key],
            graph_context=self.graph_context[key],
            glimpse_key=self.glimpse_key[key] if self.glimpse_key is not None else None,
            glimpse_val=self.glimpse_val[key] if self.glimpse_val is not None else None,
            logit_key=self.logit_key[key] if self.logit_key is not None else None,
        )

    @property
    def context_node_projected(self) -> torch.Tensor:
        """
        Alias for graph_context (backwards compatibility).

        Some decoders use the name "context_node_projected" instead of
        "graph_context". This property provides compatibility.

        Returns
        -------
        torch.Tensor
            Same as self.graph_context
        """
        return self.graph_context

    @context_node_projected.setter
    def context_node_projected(self, value: torch.Tensor):
        """Setter for context_node_projected (backwards compatibility)."""
        self.graph_context = value
