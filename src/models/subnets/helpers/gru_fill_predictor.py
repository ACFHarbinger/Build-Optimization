"""Gated Recurrent Fill Predictor."""

from typing import Tuple, cast

import torch
from torch import nn

from logic.src.models.subnets.modules import ActivationFunction


class GatedRecurrentUnitFillPredictor(nn.Module):
    """
    Predicts bin fill levels using a GRU-based recurrent network.
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        activation="relu",
        af_param=1.0,
        threshold=6.0,
        replacement_value=6.0,
        n_params=3,
        uniform_range=None,
        bidirectional=False,
    ):
        """
        Initializes the GRF Predictor.

        Args:
            input_dim: Input dimension size.
            hidden_dim: Hidden state dimension.
            num_layers: Number of GRU layers.
            dropout: Dropout probability.
            activation: Activation function for predictor MLP.
            af_param: Activation parameter.
            threshold: Activation threshold.
            replacement_value: Replacement value.
            n_params: Number of parameters.
            uniform_range: Range for uniform distribution.
            bidirectional: Whether GRU is bidirectional.
        """
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]
        super(GatedRecurrentUnitFillPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Output dimension will be doubled if bidirectional
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            ActivationFunction(
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                cast(Tuple[float, float], tuple(uniform_range)) if uniform_range else None,
            ),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fill_history):
        """
        Forward pass.

        Args:
            fill_history: Sequence of historical fill levels.

        Returns:
            Predicted next fill level.
        """
        output, hidden = self.gru(fill_history)
        last_hidden = output[:, -1, :]
        predicted_fill = self.predictor(last_hidden)
        return predicted_fill

    def init_hidden(self, batch_size, device):
        """Initializes the hidden state."""
        # (num_layers * num_directions, batch_size, hidden_dim)
        directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * directions, batch_size, self.hidden_dim).to(device)
