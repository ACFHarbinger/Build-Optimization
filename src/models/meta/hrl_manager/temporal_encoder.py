"""temporal_encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import temporal_encoder
"""

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """
    Auto-Regressive Temporal Encoder for waste history.
    """

    def __init__(self, hidden_dim: int = 64, rnn_type: str = "lstm"):
        """Initialize Class.

        Args:
            hidden_dim (int): Description of hidden_dim.
            rnn_type (str): Description of rnn_type.
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()
        # Input size is 1 because we reshape dynamic features to (Batch*N, History, 1)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

    def forward(self, dynamic: torch.Tensor) -> torch.Tensor:
        """
        Encode dynamic waste history.

        Args:
            dynamic: (Batch, N, History)

        Returns:
            temporal_embed: (Batch, N, hidden_dim)
        """
        B, N, H_len = dynamic.size()
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        dyn_flat = dynamic.view(B * N, H_len, 1)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(dyn_flat)
        else:
            _, h_n = self.rnn(dyn_flat)
        return h_n.squeeze(0).view(B, N, -1)
