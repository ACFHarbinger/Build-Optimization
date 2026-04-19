"""
Deep Decoder Policy for RL4CO.

Adapts the DeepGATDecoder-based architecture to the RL4CO architecture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.decoders.gat import DeepGATDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder


class DeepDecoderPolicy(AutoregressivePolicy):
    """
    RL4CO-style Policy using Deep Decoder architecture.
    """

    encoder: GraphAttentionEncoder
    decoder: DeepGATDecoder

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_decode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """Initialize DeepDecoderPolicy."""
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=hidden_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.decoder = DeepGATDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decode_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass executing the constructive solution generation.
        """
        # 1. Initialize embeddings
        init_embeds = self.init_embedding(td)

        # 2. Encoder
        assert self.encoder is not None, "Encoder is not initialized"
        embeddings = self.encoder(init_embeds)

        # 3. Decoder Precomputation
        assert self.decoder is not None, "Decoder is not initialized"
        fixed = self.decoder._precompute(embeddings)

        # 4. Decoding Loop
        log_likelihood: float | torch.Tensor = 0.0
        entropy: float | torch.Tensor = 0.0
        output_actions = []
        step_idx = 0
        # Assuming environment is already reset
        while not td["done"].all():
            # Wrap state
            assert self.env_name is not None, "env_name must be set"
            from logic.src.utils.data import TensorDictStateWrapper

            state_wrapper = TensorDictStateWrapper(td, self.env_name)

            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)

            # DeepDecoder output (Batch, Heads, Nodes) or (Batch, 1, Nodes)
            if logits.dim() == 3:
                logits = logits[:, 0, :] if logits.size(1) > 1 else logits.squeeze(1)

            # Invert mask (AM legacy compatibility) -> Valid Mask
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                # Teacher forcing
                action = actions[:, step_idx]
                # Compute log_prob of this action
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
                # Select action
                action, log_p, entropy_step = self._select_action(logits, valid_mask, strategy)
                entropy = entropy + entropy_step

            td["action"] = action
            td = env.step(td)["next"].clone()

            log_likelihood = log_likelihood + log_p
            output_actions.append(action)
            step_idx += 1

        reward = env.get_reward(td, torch.stack(output_actions, dim=1))

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": torch.stack(output_actions, dim=1),
            "entropy": entropy,
        }
