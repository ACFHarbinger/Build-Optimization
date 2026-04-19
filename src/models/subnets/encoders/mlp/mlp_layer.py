"""MLP layer implementation."""

from __future__ import annotations

from typing import Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.models.subnets.modules.normalization import Normalization


class MLPLayer(nn.Module):
    """
    Simple MLP layer with configurable activation and normalization.

    This layer performs: Linear → Normalization → Activation → Residual Connection.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size for the linear transformation.
    norm_config : NormalizationConfig
        Configuration for normalization layer.
    activation_config : Optional[ActivationConfig], default=None
        Configuration for activation function. If None, uses ReLU.
    """

    def __init__(
        self,
        hidden_dim: int,
        norm_config: NormalizationConfig,
        activation_config: Optional[ActivationConfig] = None,
    ):
        """
        Initialize the MLP Layer.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension size.
        norm_config : NormalizationConfig
            Normalization configuration.
        activation_config : Optional[ActivationConfig], default=None
            Activation function configuration. Defaults to ReLU if None.
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear transformation
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Normalization
        self.norm = Normalization(
            hidden_dim,
            norm_config.norm_type,
            norm_config.epsilon,
            norm_config.learn_affine,
            norm_config.track_stats,
            norm_config.momentum,
            norm_config.n_groups,
            norm_config.k_lrnorm,
        )

        # Activation function (default to ReLU for backward compatibility)
        if activation_config is None:
            activation_config = ActivationConfig(name="relu")
        self.activation = ActivationFunction(
            activation_config.name,
            activation_config.param,
            activation_config.threshold,
            activation_config.replacement_value,
            activation_config.n_params,
            (activation_config.range[0], activation_config.range[1])
            if activation_config.range and len(activation_config.range) >= 2
            else None,
        )

    def forward(self, x):
        """
        Forward pass through the MLP layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_nodes, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_nodes, hidden_dim).
        """
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        x = self.norm(x.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Apply activation
        x = self.activation(x)

        # Residual connection
        x = x_in + x

        return x
