"""Pointer Network Decoder."""

import numpy as np
import torch
from torch import nn

from .pointer_attention import PointerAttention


class PointerDecoder(nn.Module):
    """
    Standard Pointer Network Decoder.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        tanh_exploration,
        use_tanh,
        n_glimpses=1,
        mask_glimpses=True,
        mask_logits=True,
    ):
        """
        Initializes the PointerDecoder.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden dimension.
            tanh_exploration: Tanh exploration constant.
            use_tanh: Whether to use tanh.
            n_glimpses: Number of glimpses.
            mask_glimpses: Whether to mask glimpses.
            mask_logits: Whether to mask logits.
        """
        super(PointerDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.strategy = None  # Needs to be set explicitly before use

        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.pointer = PointerAttention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = PointerAttention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)

    def update_mask(self, mask, selected):
        """Updates the mask based on selected nodes."""
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):
        """
        Performs one step of recurrence.
        """
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()
        if not self.mask_logits:
            # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
            # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
            # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
            # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
            probs[logit_mask] = 0.0
            # For consistency we should also mask out in log_p, but the values set to 0 will not be sampled and
            # Therefore not be used by the reinforce estimator

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):
        """Calculates logits for the next step."""
        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)
        for _i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)

        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, eval_tours=None):
        """
        Forward pass.

        Args:
            decoder_input: The initial input to the decoder [batch_size x embed_dim].
            embedded_inputs: [sourceL x batch_size x embed_dim]
            hidden: The prev hidden state [batch_size x hidden_dim].
            context: Encoder outputs [sourceL x batch_size x hidden_dim].
            eval_tours: (Optional) tours to evaluate against.

        Returns:
            (outputs, selections), hidden
        """
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = torch.autograd.Variable(
            embedded_inputs.data.new().bool().new(embedded_inputs.size(1), embedded_inputs.size(0)).zero_(),
            requires_grad=False,
        )
        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(probs, mask) if eval_tours is None else eval_tours[:, i]

            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, TODO implement this more properly?

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:]),
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)
        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden

    def decode(self, probs, mask):
        """Decodes probabilities to actions based on strategy."""
        if self.strategy == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), (
                "Decode greedy: infeasible action has maximum probability"
            )
        elif self.strategy == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(" [!] resampling due to race condition")
                idxs = probs.multinomial().squeeze(1)
        else:
            raise AssertionError("Unknown strategy")

        return idxs
