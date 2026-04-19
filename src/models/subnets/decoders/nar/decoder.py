"""
Simple NAR Decoder: Constructs solutions using greedy/sampling from heatmaps.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.decoder import NonAutoregressiveDecoder


class SimpleNARDecoder(NonAutoregressiveDecoder):
    """
    Simple decoder for Non-Autoregressive models.

    Directly uses the heatmap as logits for selection.
    """

    def __init__(self, **kwargs):
        """Initialize Class.

        Args:
            kwargs (Any): Description of kwargs.
        """
        super().__init__(**kwargs)

    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce logits for the current step based on the heatmap.

        Args:
            td: TensorDict with current state.
            heatmap: Pre-computed heatmap [batch, n, n] or [batch, n].
            env: Environment.
            num_starts: Number of starts.

        Returns:
            Logits for selecting the next node.
        """
        # Current node
        current_node = td.get("current_node", torch.zeros(td.batch_size, dtype=torch.long, device=heatmap.device))

        if heatmap.dim() == 3:
            # Edge-based heatmap: [batch, num_nodes, num_nodes]
            # Select logits for current node
            logits = heatmap.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, heatmap.size(-1))).squeeze(1)
        else:
            # Node-based heatmap: [batch, num_nodes]
            logits = heatmap

        return logits, td["mask"]
