"""
This module contains the Meta RNN model for dynamic weight adjustment.
"""

from torch import nn


class WeightAdjustmentRNN(nn.Module):
    """
    RNN model for dynamically adjusting reward weights based on historical performance.
    This model learns patterns in how weight changes affect future performance.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the WeightAdjustmentRNN.

        Args:
            input_size (int): Size of input features (current weights + performance metrics).
            hidden_size (int): Size of RNN hidden state.
            output_size (int): Size of output (number of weights to adjust).
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        """
        super(WeightAdjustmentRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for capturing temporal patterns in weight adjustments
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Fully connected layers for weight adjustment prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize LSTM and FC layer weights for stable training"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, hidden=None):
        """
        Forward pass through the RNN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            hidden (tuple, optional): Initial hidden state. Defaults to None.

        Returns:
            tuple: (weight_adjustments, hidden)
                   - weight_adjustments: Predicted weight adjustments.
                   - hidden: Final hidden state.
        """
        lstm_out, hidden = self.lstm(x, hidden)

        # Get output from last time step
        last_output = lstm_out[:, -1, :]

        # Predict weight adjustments
        weight_adjustments = self.fc(last_output)
        return weight_adjustments, hidden
