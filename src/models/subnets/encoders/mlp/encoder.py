"""MLP Encoder."""

from __future__ import annotations

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig

from .mlp_layer import MLPLayer


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with configurable activation and normalization.

    This encoder consists of stacked MLPLayer instances, each performing:
    Linear → Normalization → Activation → Residual Connection.

    Unlike graph-based encoders, this encoder operates independently of graph
    structure and can be used for simple feature transformations.

    Parameters
    ----------
    n_layers : int
        Number of MLP layers to stack.
    feed_forward_hidden : int
        Hidden dimension size for each layer.
    norm : str, default="layer"
        Normalization type: "batch", "layer", "instance", "group", or None.
    learn_affine : bool, default=True
        Whether to learn affine parameters in normalization layers.
    track_norm : bool, default=False
        Whether to track running statistics in normalization.
    activation : str, default="relu"
        Activation function name. Defaults to "relu" for backward compatibility.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_layers: int,
        feed_forward_hidden: int,
        norm: str = "layer",
        learn_affine: bool = True,
        track_norm: bool = False,
        activation: str = "relu",
        *args,
        **kwargs,
    ):
        """
        Initialize the MLP Encoder.

        Parameters
        ----------
        n_layers : int
            Number of layers.
        feed_forward_hidden : int
            Hidden dimension.
        norm : str, default="layer"
            Normalization type.
        learn_affine : bool, default=True
            Learn affine parameters in normalization.
        track_norm : bool, default=False
            Track running statistics.
        activation : str, default="relu"
            Activation function name.
        """
        super(MLPEncoder, self).__init__()

        # Create normalization config
        self.norm_config = NormalizationConfig(
            norm_type=norm,
            learn_affine=learn_affine,
            track_stats=track_norm,
        )

        # Create activation config
        self.activation_config = ActivationConfig(name=activation)

        # Store configuration
        self.n_layers = n_layers
        self.hidden_dim = feed_forward_hidden

        # Build layer stack
        self.layers = nn.ModuleList(
            [
                MLPLayer(
                    feed_forward_hidden,
                    self.norm_config,
                    self.activation_config,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, graph=None) -> torch.Tensor:
        """
        Forward pass through all MLP layers.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (batch_size, num_nodes, hidden_dim).
        graph : Optional
            Graph structure (unused by MLP encoder, included for API compatibility).

        Returns
        -------
        torch.Tensor
            Transformed node features of shape (batch_size, num_nodes, hidden_dim).
        """
        # Note: graph parameter is unused but kept for API compatibility with other encoders
        for layer in self.layers:
            x = layer(x)

        return x
