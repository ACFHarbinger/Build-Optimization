"""NARGNN Encoder base implementation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.non_autoregressive.encoder import NonAutoregressiveEncoder
from logic.src.models.subnets.embeddings import get_init_embedding

from .edge_embedding import SimplifiedEdgeEmbedding
from .edge_heatmap_generator import EdgeHeatmapGenerator
from .gnn_encoder import SimplifiedGNNEncoder


class NARGNNEncoder(NonAutoregressiveEncoder):
    """
    Anisotropic Graph Neural Network encoder with edge-gating mechanism.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn: str = "silu",
        agg_fn: str = "mean",
        linear_bias: bool = True,
        k_sparse: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            env_name (str): Description of env_name.
            init_embedding (Optional[nn.Module]): Description of init_embedding.
            edge_embedding (Optional[nn.Module]): Description of edge_embedding.
            graph_network (Optional[nn.Module]): Description of graph_network.
            heatmap_generator (Optional[nn.Module]): Description of heatmap_generator.
            num_layers_heatmap_generator (int): Description of num_layers_heatmap_generator.
            num_layers_graph_encoder (int): Description of num_layers_graph_encoder.
            act_fn (str): Description of act_fn.
            agg_fn (str): Description of agg_fn.
            linear_bias (bool): Description of linear_bias.
            k_sparse (Optional[int]): Description of k_sparse.
            kwargs (Any): Description of kwargs.
        """
        super().__init__()
        self.env_name = env_name

        self.init_embedding = get_init_embedding(self.env_name, embed_dim) if init_embedding is None else init_embedding

        self.edge_embedding = (
            SimplifiedEdgeEmbedding(embed_dim=embed_dim, k_sparse=k_sparse, linear_bias=linear_bias)
            if edge_embedding is None
            else edge_embedding
        )

        self.graph_network = (
            SimplifiedGNNEncoder(
                embed_dim=embed_dim,
                num_layers=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
            )
            if graph_network is None
            else graph_network
        )

        self.heatmap_generator = (
            EdgeHeatmapGenerator(
                embed_dim=embed_dim,
                num_layers=num_layers_heatmap_generator,
                linear_bias=linear_bias,
            )
            if heatmap_generator is None
            else heatmap_generator
        )

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """Forward pass."""
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        heatmap_logits = self.heatmap_generator(graph)
        return heatmap_logits, node_embed
