"""LSTM Recurrent Fill Predictor."""

from typing import Tuple, cast

from torch import nn

from logic.src.models.subnets.modules import ActivationFunction


class LongShortTermMemoryFillPredictor(nn.Module):
    """
    Predicts bin fill levels using an LSTM-based recurrent network.
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
        Initializes the LSTM Fill Predictor.

        Args:
            input_dim: Input dimension size.
            hidden_dim: Hidden state dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            activation: Activation function for predictor MLP.
            af_param: Activation parameter.
            threshold: Activation threshold.
            replacement_value: Replacement value.
            n_params: Number of parameters.
            uniform_range: Range for uniform distribution.
            bidirectional: Whether LSTM is bidirectional.
        """
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]
        super(LongShortTermMemoryFillPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
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
        # LSTM returns output, (h_n, c_n)
        output, (h_n, c_n) = self.lstm(fill_history)

        # Taking the last time step's output for prediction
        last_output = output[:, -1, :]
        predicted_fill = self.predictor(last_output)
        return predicted_fill

    def get_embedding(self, fill_history):
        """
        Get the LSTM embedding (hidden state) for the sequence.
        Useful for downstream tasks like the Manager agent.

        Args:
            fill_history: Sequence of historical fill levels.

        Returns:
            Embedding tensor (Batch, Hidden).
        """
        output, (h_n, c_n) = self.lstm(fill_history)

        # If bidirectional, we might want to concatenate the last hidden states of both directions
        # But for simplicity and compatibility with Manager which usually expects (Batch, Hidden),
        # we can use the last output timestep which contains the hidden state of the last layer.

        # output is (Batch, Seq, NumDirections * Hidden)
        return output[:, -1, :]
