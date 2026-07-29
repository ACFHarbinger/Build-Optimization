"""NARGNN Node Encoder variant."""

from __future__ import annotations

from typing import Tuple

import torch
from tensordict import TensorDict

from .encoder import NARGNNEncoder


class NARGNNNodeEncoder(NARGNNEncoder):
    """
    NARGNN encoder variant that returns node embeddings instead of heatmaps.
    """

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Forward pass."""
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        proc_embeds = graph.x
        batch_size = node_embed.shape[0]
        proc_embeds = proc_embeds.reshape(batch_size, -1, proc_embeds.shape[1])
        return proc_embeds, node_embed
