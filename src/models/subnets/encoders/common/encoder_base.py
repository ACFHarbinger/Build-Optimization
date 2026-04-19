"""
Base encoder classes for reducing boilerplate across encoder implementations.

This module provides abstract base classes that handle common encoder patterns:
- Parameter initialization (n_heads, embed_dim, n_layers, normalization, activation)
- Layer stacking via ModuleList
- Standard forward pass with dropout
- Optional hyper-connection support

Subclasses only need to implement `_create_layer()` to define their specific layer type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig


class TransformerEncoderBase(nn.Module, ABC):
    """
    Abstract base class for transformer-style graph encoders.

    Handles common boilerplate:
    - Parameter validation and initialization
    - Layer stacking via `_create_layer()` abstract method
    - Standard forward pass pattern with dropout
    - Optional hyper-connection expansion/collapse

    Usage Example
    -------------
    ```python
    class MyCustomEncoder(TransformerEncoderBase):
        def _create_layer(self, layer_idx: int) -> nn.Module:
            return MyCustomLayer(
                n_heads=self.n_heads,
                embed_dim=self.embed_dim,
                feed_forward_hidden=self.feed_forward_hidden,
                norm_config=self.norm_config,
                activation_config=self.activation_config,
            )
    ```

    Parameters
    ----------
    n_heads : int
        Number of attention heads (for multi-head attention layers).
    embed_dim : int
        Embedding dimension (must be divisible by n_heads for attention).
    n_layers : int
        Number of encoder layers to stack.
    feed_forward_hidden : int, default=512
        Hidden dimension for feed-forward sublayers.
    norm_config : Optional[NormalizationConfig], default=None
        Normalization configuration. Defaults to batch normalization if None.
    activation_config : Optional[ActivationConfig], default=None
        Activation function configuration. Defaults to ReLU if None.
    dropout_rate : float, default=0.1
        Dropout probability applied after encoder layers.
    connection_type : str, default="skip"
        Connection type: "skip" (residual), "dense", "hyper" (hyper-connections).
    expansion_rate : int, default=4
        Expansion factor for hyper-connections (only used if connection_type="hyper*").
    **kwargs
        Additional keyword arguments passed to subclass `_create_layer()`.

    Attributes
    ----------
    layers : nn.ModuleList
        List of encoder layers created by `_create_layer()`.
    dropout : nn.Dropout
        Dropout layer applied after encoding.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the base encoder with common parameters."""
        super(TransformerEncoderBase, self).__init__()

        # Store parameters for subclass access
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.feed_forward_hidden = feed_forward_hidden
        self.dropout_rate = dropout_rate
        self.conn_type = connection_type
        self.expansion_rate = expansion_rate
        self.kwargs = kwargs

        # Default configs if not provided
        self.norm_config = norm_config if norm_config is not None else NormalizationConfig()
        self.activation_config = activation_config if activation_config is not None else ActivationConfig()

        # Validate embed_dim divisibility for multi-head attention
        if n_heads > 0 and embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}). "
                f"Got remainder: {embed_dim % n_heads}"
            )

        # Create layers via abstract method
        self.layers = nn.ModuleList([self._create_layer(layer_idx) for layer_idx in range(n_layers)])

        # Standard dropout
        self.dropout = nn.Dropout(dropout_rate)

    @abstractmethod
    def _create_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a single encoder layer.

        Subclasses must implement this to define their specific layer architecture.
        The layer should accept (x, mask/edges) in forward() and return updated x.

        Parameters
        ----------
        layer_idx : int
            Index of the layer being created (0 to n_layers-1).
            Can be used for layer-specific configurations.

        Returns
        -------
        nn.Module
            The created encoder layer.

        Example
        -------
        ```python
        def _create_layer(self, layer_idx: int) -> nn.Module:
            return GATMultiHeadAttentionLayer(
                n_heads=self.n_heads,
                embed_dim=self.embed_dim,
                feed_forward_hidden=self.feed_forward_hidden,
                norm_config=self.norm_config,
                activation_config=self.activation_config,
            )
        ```
        """
        raise NotImplementedError("Subclasses must implement _create_layer()")

    def forward(
        self,
        x: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard encoder forward pass with layer stacking.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (batch_size, num_nodes, embed_dim).
        edges : Optional[torch.Tensor], default=None
            Edge information (adjacency matrix, edge indices, or edge features).
            Format depends on the specific encoder implementation.
        mask : Optional[torch.Tensor], default=None
            Attention mask of shape (batch_size, num_nodes, num_nodes).
            If None, all nodes attend to all nodes.

        Returns
        -------
        torch.Tensor
            Encoded node features of shape (batch_size, num_nodes, embed_dim).

        Notes
        -----
        - Supports hyper-connections if connection_type contains "hyper"
        - Applies dropout after all layers
        - Preserves input shape (batch, nodes, embed_dim)
        """
        # 1. Expand input for hyper-connections if needed
        if "hyper" in self.conn_type:
            # x: (B, N, D) -> H: (B, N, D, expansion_rate)
            H = x.unsqueeze(-1).repeat(1, 1, 1, self.expansion_rate)
            curr = H
        else:
            curr = x

        # 2. Pass through encoder layers
        # Use either edges or mask parameter (layer-dependent)
        for layer in self.layers:
            if edges is not None:
                curr = layer(curr, edges=edges)
            elif mask is not None:
                curr = layer(curr, mask=mask)
            else:
                curr = layer(curr)

        # 3. Collapse hyper-connections if needed
        if "hyper" in self.conn_type:
            curr = curr.mean(dim=-1)  # (B, N, D, R) -> (B, N, D)

        # 4. Apply dropout
        output: torch.Tensor = self.dropout(curr)
        return output  # (batch_size, num_nodes, embed_dim)
