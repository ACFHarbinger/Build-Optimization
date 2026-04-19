from __future__ import annotations

from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class ImprovementStepDecoder(ImprovementDecoder):
    """
    Decoder that selects a vectorized operator to improve the solution.
    Output is a probability distribution over available operators.
    """

    def __init__(self, embed_dim: int = 128, n_operators: int = 6, hidden_dim: int = 128):
        super().__init__(embed_dim=embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_operators)
        )
        self.context_projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict operator selection logits.

        Args:
            td: TensorDict containing current state/solution.
            embeddings: Graph embeddings [Batch, Nodes, Embed].
            env: Environment.

        Returns:
            Tuple of (logits, action_mask).
        """
        # Simple Global Context: Mean pooling of node embeddings
        # Real implementation might encode current tour state more deeply
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        graph_context = embeddings.mean(dim=1)  # [B, Embed]

        # Project and predict
        context = self.context_projection(graph_context)
        logits = self.output_head(context)  # [B, n_operators]

        # For now, no mask (all operators always available)
        mask = torch.zeros_like(logits, dtype=torch.bool)

        return logits, mask
