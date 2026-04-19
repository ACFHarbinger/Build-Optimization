"""
Temporal Attention Model Policy.
"""

from typing import Optional, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.subnets.helpers.gru_fill_predictor import GatedRecurrentUnitFillPredictor
from logic.src.models.subnets.helpers.lstm_fill_predictor import LongShortTermMemoryFillPredictor
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper


class TemporalAMPolicy(AttentionModelPolicy):
    """
    Temporal Attention Model Policy.

    Incorporates historical waste data and fill level predictions using
    a Gated Recurrent Fill Predictor.
    """

    fill_predictor: Union[LongShortTermMemoryFillPredictor, GatedRecurrentUnitFillPredictor]
    temporal_embed: nn.Linear
    combine_embeddings: nn.Sequential

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        temporal_horizon: int = 5,
        predictor_layers: int = 2,
        predictor_type: str = "gru",
        **kwargs,
    ):
        """Initialize TemporalAMPolicy."""
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        self.temporal_horizon = temporal_horizon

        # Sub-components for temporal features
        if predictor_type == "lstm":
            self.fill_predictor = LongShortTermMemoryFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=kwargs.get("dropout_rate", 0.1),
            )
        else:
            self.fill_predictor = GatedRecurrentUnitFillPredictor(  # type: ignore[assignment]
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=kwargs.get("dropout_rate", 0.1),
            )

        self.temporal_embed = nn.Linear(1, embed_dim)

        # Combiner for base and temporal embeddings
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            ActivationFunction(kwargs.get("activation_function", "gelu")),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass with temporal feature prediction.
        """
        # 1. Prepare temporal features (fill_history)
        if "fill_history" not in td.keys():
            batch_size = td.batch_size[0]
            num_nodes = td["locs"].shape[1]
            td["fill_history"] = torch.zeros(
                (batch_size, num_nodes, self.temporal_horizon, 1),
                device=td.device,
            )

        fill_history = td["fill_history"]  # (batch, nodes, horizon, 1)
        batch_size, num_nodes, horizon, _ = fill_history.size()

        # Predict future fills
        # Reshape to (batch * nodes, horizon, 1) for GRU
        h_flat = fill_history.view(batch_size * num_nodes, horizon, 1)
        predicted_fills = self.fill_predictor(h_flat)  # (batch * nodes, 1)
        predicted_fills = predicted_fills.view(batch_size, num_nodes, 1)

        # 2. Get base embeddings
        init_embeds = self.init_embedding(td)  # (batch, nodes, embed_dim)

        # 3. Create temporal embeddings
        fill_embeds = self.temporal_embed(predicted_fills)  # (batch, nodes, embed_dim)

        # 4. Combine
        combined_embeds = self.combine_embeddings(torch.cat([init_embeds, fill_embeds], dim=-1))

        # 2. Encoder
        edges = td.get("edges", None)
        assert self.encoder is not None, "Encoder is not initialized"
        embeddings = self.encoder(combined_embeds, edges)

        # 3. Decoder Precomputation
        assert self.decoder is not None, "Decoder is not initialized"
        fixed = self.decoder._precompute(embeddings)

        # Start the actual training loop
        log_likelihood: Union[torch.Tensor, float] = 0.0
        entropy: Union[torch.Tensor, float] = 0.0
        output_actions = []
        step_idx = 0
        # Assuming environment is already reset
        while not td["done"].all():
            # Wrap state
            assert self.env_name is not None, "env_name must be set"
            state_wrapper = TensorDictStateWrapper(td, self.env_name)
            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)
            logits = logits[:, 0, :]

            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                action = actions[:, step_idx]
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
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
