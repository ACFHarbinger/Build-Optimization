"""Pointer attention module."""

import math

import torch
from torch import nn


class PointerAttention(nn.Module):
    """A generic attention module for a decoder in seq2seq."""

    def __init__(self, dim, use_tanh=False, C=10):
        """
        Initializes PointerAttention.

        Args:
            dim: Dimension of attention vector.
            use_tanh: Whether to use tanh exploration.
            C: Tanh exploration constant.
        """
        super(PointerAttention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Calculate attention logits.

        Args:
            query: Hidden state of the decoder at the current time step (batch x dim).
            ref: Set of hidden states from the encoder (sourceL x batch x hidden_dim).

        Returns:
            Projected reference and logits.
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        logits = self.C * self.tanh(u) if self.use_tanh else u
        return e, logits
