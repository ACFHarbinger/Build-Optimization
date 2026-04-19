"""Graph Convolution Encoder."""

import torch
from torch import nn

from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules import GatedGraphConvolution


class GraphConvolutionEncoder(nn.Module):
    """
    Encoder based on Gated Graph Convolutions.

    Uses stacked GatedGraphConvolution layers for node and edge feature updates.
    Unlike transformer-based encoders, this encoder operates on graph structure
    directly using graph convolution operations.

    Parameters
    ----------
    n_layers : int
        Number of graph convolution layers.
    feed_forward_hidden : int
        Hidden dimension for node and edge features.
    agg : str, default="sum"
        Aggregation method for neighbor features: "sum", "mean", or "max".
    norm : str, default="layer"
        Normalization type: "batch", "layer", "instance", or "group".
    learn_affine : bool, default=True
        Whether to learn affine parameters in normalization.
    track_norm : bool, default=False
        Whether to track running statistics in normalization.
    gated : bool, default=True
        Whether to use gated update mechanism in graph convolutions.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_layers,
        feed_forward_hidden,
        agg="sum",
        norm="layer",
        learn_affine=True,
        track_norm=False,
        gated=True,
        *args,
        **kwargs,
    ):
        """Initialize the GCN Encoder."""
        super(GraphConvolutionEncoder, self).__init__()

        # Create normalization config for consistency with other encoders
        self.norm_config = NormalizationConfig(
            norm_type=norm,
            learn_affine=learn_affine,
            track_stats=track_norm,
        )

        # Store layer configuration
        self.n_layers = n_layers
        self.hidden_dim = feed_forward_hidden
        self.agg = agg
        self.gated = gated

        # Edge embedding: binary edge features (0/1) -> hidden_dim
        self.init_embed_edges = nn.Embedding(2, feed_forward_hidden)

        # Stack of gated graph convolution layers
        self.layers = nn.ModuleList(
            [
                GatedGraphConvolution(
                    hidden_dim=feed_forward_hidden,
                    aggregation=agg,
                    norm=norm,
                    learn_affine=learn_affine,
                    gated=gated,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, edges):
        """
        Forward pass through graph convolution layers.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (batch_size, num_nodes, hidden_dim).
        edges : torch.Tensor
            Graph adjacency information. Can be:
            - Dense: (batch_size, num_nodes, num_nodes) - adjacency matrix
            - Sparse: (batch_size, 2, num_edges) - edge index list

        Returns
        -------
        torch.Tensor
            Updated node features of shape (batch_size, num_nodes, hidden_dim).

        Notes
        -----
        If edges are provided in sparse format (edge index list), they are
        converted to dense adjacency matrices before processing.
        """
        batch_size, num_nodes, _ = x.size()

        # Convert sparse edge list to dense adjacency matrix if needed
        if edges.dim() == 3 and edges.size(1) == 2:
            adj = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device, dtype=torch.long)
            # Build adjacency matrix from edge indices
            for b in range(batch_size):
                # Ensure indices are long integers
                idx = edges[b].long()
                # Filter out-of-bounds indices
                valid_mask = (idx[0] < num_nodes) & (idx[1] < num_nodes)
                src = idx[0][valid_mask]
                dst = idx[1][valid_mask]
                adj[b, src, dst] = 1
            edges = adj

        # Embed edge features: {0, 1} -> hidden_dim
        edge_embed = self.init_embed_edges(edges.type(torch.long))

        # Pass through graph convolution layers
        for layer in self.layers:
            x, edge_embed = layer(x, edge_embed, edges)

        return x
