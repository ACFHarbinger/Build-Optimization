"""Gated Graph Attention Convolution Encoder."""

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .attention_gated_convolution_layer import AttentionGatedConvolutionLayer


class GatedGraphAttConvEncoder(TransformerEncoderBase):
    """
    Gated Graph Attention Convolution Encoder using stacked AttentionGatedConvolutionLayers.

    This encoder combines graph attention mechanisms with gated convolutions and supports
    edge embeddings derived from distance matrices.

    Inherits from TransformerEncoderBase for common encoder patterns like layer stacking,
    dropout, and configuration management.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    embed_dim : int
        Embedding dimension.
    n_layers : int
        Number of encoder layers.
    n_sublayers : Optional[int], default=None
        Number of sublayers (unused, kept for API compatibility).
    feed_forward_hidden : int, default=512
        Hidden dimension for feed-forward layers.
    normalization : str, default="batch"
        Normalization type: "batch", "layer", "instance", or "group".
    epsilon_alpha : float, default=1e-05
        Epsilon for normalization stability.
    learn_affine : bool, default=True
        Whether to learn affine parameters in normalization.
    track_stats : bool, default=False
        Whether to track running stats in batch norm.
    momentum_beta : float, default=0.1
        Momentum for batch norm running stats.
    locresp_k : float, default=1.0
        Local response normalization parameter.
    n_groups : int, default=3
        Number of groups for group normalization.
    activation : str, default="gelu"
        Activation function name.
    af_param : float, default=1.0
        Activation function parameter.
    threshold : float, default=6.0
        Activation threshold.
    replacement_value : float, default=6.0
        Activation replacement value.
    n_params : int, default=3
        Number of activation parameters.
    uniform_range : list, default=[0.125, 1/3]
        Uniform range for activation.
    dropout_rate : float, default=0.1
        Dropout probability.
    agg : str, default="sum"
        Aggregation method for graph convolutions: "sum", "mean", or "max".
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
        epsilon_alpha: float = 1e-05,
        learn_affine: bool = True,
        track_stats: bool = False,
        momentum_beta: float = 0.1,
        locresp_k: float = 1.0,
        n_groups: int = 3,
        activation: str = "gelu",
        af_param: float = 1.0,
        threshold: float = 6.0,
        replacement_value: float = 6.0,
        n_params: int = 3,
        uniform_range: Optional[list] = None,
        dropout_rate: float = 0.1,
        agg: str = "sum",
        **kwargs,
    ) -> None:
        """Initialize the GatedGraphAttConvEncoder."""
        # Store GGAC-specific parameters BEFORE calling super().__init__()
        # because _create_layer() is called during base class initialization
        self.aggregate = agg
        self.n_sublayers = n_sublayers

        # Create config objects from individual parameters
        norm_config = NormalizationConfig(
            norm_type=normalization,
            epsilon=epsilon_alpha,
            learn_affine=learn_affine,
            track_stats=track_stats,
            momentum=momentum_beta,
            k_lrnorm=locresp_k,
            n_groups=n_groups,
        )

        activation_config = ActivationConfig(
            name=activation,
            param=af_param,
            threshold=threshold,
            replacement_value=replacement_value,
            n_params=n_params,
            range=uniform_range if uniform_range is not None else [0.125, 1 / 3],
        )

        # Initialize base class (handles layer creation, dropout, default configs)
        # Note: This will call _create_layer() which needs self.aggregate to be set
        super(GatedGraphAttConvEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # Initial Edge Embedding (Distance -> Embed)
        # Set after base class init to ensure all parameters are available
        self.dist_norm = nn.BatchNorm1d(1)  # Normalize raw distances
        self.init_edge_embed = nn.Linear(1, embed_dim)

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a single AttentionGatedConvolutionLayer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer being created (0 to n_layers-1).
            Unused by GGAC but required by base class interface.

        Returns
        -------
        nn.Module
            AttentionGatedConvolutionLayer instance.
        """
        # Note: layer_idx is unused but required by base class signature
        _ = layer_idx  # Explicitly mark as unused
        return AttentionGatedConvolutionLayer(
            self.n_heads,
            self.embed_dim,
            self.feed_forward_hidden,
            self.norm_config.norm_type,
            self.norm_config.epsilon,
            self.norm_config.learn_affine,
            self.norm_config.track_stats,
            self.norm_config.momentum,
            self.norm_config.k_lrnorm,
            self.norm_config.n_groups,
            self.activation_config.name,
            self.activation_config.param,
            self.activation_config.threshold,
            self.activation_config.replacement_value,
            self.activation_config.n_params,
            self.activation_config.range,
            gated=True,
            agg=self.aggregate,
        )

    def forward(
        self,
        x: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with edge embedding support.

        This encoder supports distance-based edge embeddings, which are initialized
        from a distance matrix and updated through the layers alongside node embeddings.

        GGAC-specific override: Unlike the base class forward, this method handles
        both node and edge embeddings, with edge embeddings derived from distance matrices.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (batch_size, num_nodes, embed_dim).
        edges : Optional[torch.Tensor], default=None
            Edge information (unused in GGAC, kept for compatibility).
        mask : Optional[torch.Tensor], default=None
            Boolean adjacency mask of shape (batch_size, num_nodes, num_nodes).
            If None, assumes a fully connected graph.
        dist : Optional[torch.Tensor], default=None
            Distance matrix of shape (batch_size, num_nodes, num_nodes) or
            (num_nodes, num_nodes). If provided, used to initialize edge embeddings.
            This is a GGAC-specific parameter not present in the base class.

        Returns
        -------
        torch.Tensor
            Encoded node features of shape (batch_size, num_nodes, embed_dim).

        Notes
        -----
        - Edge embeddings are derived from distance matrices via normalization and linear projection
        - If no distance matrix provided, edge embeddings are initialized to zeros
        - The encoder updates both node and edge embeddings through layers, but only returns nodes
        - The `edges` parameter is kept for API compatibility but `mask` is used for attention
        """
        # Use mask parameter (edges is kept for base class compatibility but not used)
        adjacency_mask = mask

        # Initialize edge embeddings from distance matrix
        if dist is None:
            batch_size, num_nodes, _ = x.size()
            e = torch.zeros(batch_size, num_nodes, num_nodes, self.embed_dim, device=x.device)
        else:
            # Ensure dist has proper shape: (B, N1, N2, 1)
            if len(dist.shape) == 2:
                dist = dist.unsqueeze(0).unsqueeze(-1)
            elif len(dist.shape) == 3:
                dist = dist.unsqueeze(-1)

            B, N1, N2, _ = dist.shape
            # Normalize distances and project to embedding space
            dist_flat = dist.view(-1, 1)
            dist_norm = self.dist_norm(dist_flat)
            e = self.init_edge_embed(dist_norm.view(B, N1, N2, 1))

        # Create default mask if not provided (fully connected)
        if adjacency_mask is None:
            batch_size, num_nodes, _ = x.size()
            adjacency_mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=x.device)

        # Pass through encoder layers (updates both x and e)
        for layer in self.layers:
            x, e = layer(x, e, mask=adjacency_mask)

        # Apply dropout and return node embeddings
        return self.dropout(x)
