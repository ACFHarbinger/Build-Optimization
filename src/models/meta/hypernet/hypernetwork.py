"""hypernetwork.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import hypernetwork
"""

import torch
from torch import nn

from logic.src.models.subnets.modules import ActivationFunction, Normalization


class HyperNetwork(nn.Module):
    """
    HyperNetwork that generates adaptive cost weights for time-variant RL tasks.
    Takes temporal and performance metrics as input and outputs cost weights.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_days=365,
        embed_dim=16,
        hidden_dim=64,
        normalization="layer",
        activation="relu",
        learn_affine=True,
        bias=True,
    ):
        """
        Initialize the HyperNetwork.

        Args:
            input_dim (int): Dimension of input metrics.
            output_dim (int): Dimension of output weights.
            n_days (int, optional): Number of days in the year. Defaults to 365.
            embed_dim (int, optional): Dimension of time embedding. Defaults to 16.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
            normalization (str, optional): Normalization type. Defaults to 'layer'.
            activation (str, optional): Activation function. Defaults to 'relu'.
            learn_affine (bool, optional): Whether to learn affine parameters in norm. Defaults to True.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        """
        super(HyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_days = n_days
        self.time_embedding = nn.Embedding(n_days, embed_dim)

        # Combined input: metrics + time embedding
        combined_dim = input_dim + embed_dim

        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

        # Output activation to ensure positive weights
        self.activation = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for the linear layers using Xavier Uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metrics, day):
        """
        Generate cost weights based on current metrics and temporal information

        Args:
            metrics: Tensor of performance metrics [batch_size, input_dim]
            day: Tensor of day indices [batch_size]

        Returns:
            cost_weights: Tensor of generated cost weights [batch_size, output_dim]
        """
        # Get time embeddings
        day_embed = self.time_embedding(day % self.n_days)

        # Concatenate metrics with time embeddings
        combined = torch.cat([metrics, day_embed], dim=1)

        # Generate raw weights
        raw_weights = self.layers(combined)

        # Apply activation and normalize to sum to constraint
        weights = self.activation(raw_weights)

        return weights
