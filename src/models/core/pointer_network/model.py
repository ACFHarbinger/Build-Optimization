"""
This module contains the Pointer Network model implementation.
"""

import math

import torch
from torch import nn

from logic.src.models.subnets import PointerDecoder, PointerEncoder


# Attention, Learn to Solve Routing Problems
class PointerNetwork(nn.Module):
    """
    Pointer Network implementing the Attention Model logic for VRP.

    References:
        Vinyals et al. (2015) - Pointer Networks.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        problem,
        n_encode_layers=None,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization=None,
        **kwargs,
    ):
        """
        Initialize the Pointer Network.

        Args:
            embed_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the hidden layers.
            problem (object): The problem instance wrapper.
            n_encode_layers (int, optional): Number of encoder layers. Defaults to None.
            tanh_clipping (float, optional): Tanh clipping value. Defaults to 10.0.
            mask_inner (bool, optional): Whether to mask inner attention. Defaults to True.
            mask_logits (bool, optional): Whether to mask logits. Defaults to True.
            normalization (str, optional): Normalization type. Defaults to None.
            **kwargs: Arbitrary keyword arguments.
        """
        super(PointerNetwork, self).__init__()
        self.problem = problem
        self.input_dim = 2
        self.encoder = PointerEncoder(embed_dim, hidden_dim)
        self.decoder = PointerDecoder(
            embed_dim,
            hidden_dim,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        # Trainable initial hidden states
        std = 1.0 / math.sqrt(embed_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embed_dim))
        self.decoder_in_0.data.uniform_(-std, std)

        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embed_dim))
        self.embedding.data.uniform_(-std, std)

    def set_strategy(self, strategy):
        """
        Set the decoding strategy for the model.

        Args:
            strategy (str): The decoding strategy ('greedy' or 'sampling').
        """
        self.decoder.strategy = strategy

    def forward(self, inputs, eval_tours=None, return_pi=False):
        """
        Forward pass of the Pointer Network.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, graph_size, input_dim].
            eval_tours (torch.Tensor, optional): Tours for evaluation/Teacher Forcing. Defaults to None.
            return_pi (bool, optional): Whether to return the action sequence. Defaults to False.

        Returns:
            tuple: (cost, log_likelihood, [pi])
        """
        batch_size, graph_size, input_dim = inputs.size()
        embedded_inputs = torch.mm(inputs.transpose(0, 1).contiguous().view(-1, input_dim), self.embedding).view(
            graph_size, batch_size, -1
        )

        # query the actor net for the input indices
        # making up the output, and the pointer attn
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        cost, mask = self.problem.get_costs(inputs, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):
        """calc log likelihood.

        Args:
            _log_p (Any): Description of _log_p.
            a (Any): Description of a.
            mask (Any): Description of mask.

        Returns:
            Any: Description of return value.
        """
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, inputs, eval_tours=None):
        """inner.

        Args:
            inputs (Any): Description of inputs.
            eval_tours (Any): Description of eval_tours.

        Returns:
            Any: Description of return value.
        """
        encoder_hx = encoder_cx = torch.autograd.Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()),
            requires_grad=False,
        )

        # Encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # Repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(
            decoder_input, inputs, dec_init_state, enc_h, eval_tours
        )
        return pointer_probs, input_idxs
