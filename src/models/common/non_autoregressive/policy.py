"""nonautoregressive_policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import nonautoregressive_policy
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .decoder import NonAutoregressiveDecoder
from .encoder import NonAutoregressiveEncoder


class NonAutoregressivePolicy(nn.Module, ABC):
    """
    Base class for non-autoregressive policies.

    Combines a NAR encoder (heatmap prediction) with a NAR decoder
    (solution construction) to form a complete policy.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize NonAutoregressivePolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode heatmap + decode solution.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions and reward calculation.
            num_starts: Number of solution constructions (for stochastic decoders).
            **kwargs: Additional arguments for encoder/decoder.

        Returns:
            Dictionary containing:
                - reward: Final reward/cost [batch] or [batch, num_starts]
                - log_likelihood: Log probability of solution [batch]
                - actions: Solution sequence [batch, seq_len]
                - heatmap: Predicted heatmap from encoder
        """
        # Encode: predict heatmap
        encoder_out = self.encoder(td, **kwargs) if self.encoder is not None else None
        heatmap = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

        # Decode: construct solution(s) from heatmap
        if self.decoder is not None and heatmap is not None:
            out = self.decoder(td, heatmap, env, num_starts=num_starts, **kwargs)
        else:
            # Fallback for subclasses that override forward entirely
            out = {}

        out["heatmap"] = heatmap
        return out

    def set_strategy(self, strategy: str, **kwargs):
        """Set decoding strategy (compatibility with evaluation pipeline)."""
        self._strategy = strategy
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    def common_decoding(
        self,
        strategy: str,
        td: TensorDict,
        env: RL4COEnvBase,
        heatmap: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **decoding_kwargs,
    ):
        """
        Common decoding logic for NAR models.

        Args:
            strategy: Decoding strategy ('sampling', 'greedy', etc.)
            td: Initial TensorDict
            env: Environment
            heatmap: Predicted heatmap from encoder
            actions: Pre-specified actions for evaluation
            **decoding_kwargs: Additional arguments for decoding

        Returns:
            Tuple of (logprobs, actions, td, env)
        """
        if actions is not None:
            strategy = "evaluate"
            decoding_kwargs["actions"] = actions

        from logic.src.utils.decoding import batchify, get_decoding_strategy

        strategy_obj = get_decoding_strategy(strategy, **decoding_kwargs)

        # Pre-decoding hook
        td, env, num_starts_hook = strategy_obj.pre_decoder_hook(td, env)

        # Determine num_starts
        num_starts = decoding_kwargs.get("num_starts", decoding_kwargs.get("num_samples", num_starts_hook))
        if num_starts is None:
            num_starts = 1
        num_starts = int(num_starts)

        # Update heatmap and td to match num_starts if needed
        if num_starts > 1:
            if td.size(0) != heatmap.size(0) * num_starts and td.size(0) == heatmap.size(0):
                # This might happen if pre_decoder_hook didn't batchify
                td = batchify(td, num_starts)

            if heatmap.size(0) != td.size(0):
                heatmap = batchify(heatmap, num_starts)

        if self.decoder is None:
            raise ValueError("common_decoding requires a decoder to be set")

        # Main decoding loop
        step = 0
        actions_list = []
        log_probs_list = []
        while not td["done"].all():
            # In NAR models, the decoder uses the pre-computed heatmap
            logits, mask = self.decoder(td, heatmap, env)
            action, log_prob, _ = strategy_obj.step(logits, mask, td)

            actions_list.append(action)
            log_probs_list.append(log_prob)

            td.set("action", action)
            td.update(env.step(td)["next"])
            step += 1

        actions = torch.stack(actions_list, dim=1)
        logprobs = torch.stack(log_probs_list, dim=1)

        # Post-decoding hook
        logprobs, actions, td, env = strategy_obj.post_decoder_hook(td, env, logprobs, actions)
        return logprobs, actions, td, env

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self
