"""Pointer Network Encoder."""

import math

import torch
from torch import nn


class PointerEncoder(nn.Module):
    """
    Maps a graph represented as an input sequence to a hidden vector.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the PointerEncoder.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
        """
        super(PointerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        """
        Forward pass.

        Args:
            x: Input sequence.
            hidden: Initial hidden state.
        """
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1.0 / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx
