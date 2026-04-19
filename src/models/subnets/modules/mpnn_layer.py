"""Message Passing Layer for MPNN."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.modules.normalization import Normalization

if TYPE_CHECKING:
    # This block is only seen by static type checkers
    from torch_geometric.nn import MessagePassing
else:
    # This block is executed at runtime
    try:
        from torch_geometric.nn import MessagePassing
    except (ImportError, OSError):
        MessagePassing = object


class MessagePassingLayer(MessagePassing):
    """
    Message Passing Layer with custom node and edge update models.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
        bias: bool = True,
    ):
        """
        Initialize MessagePassingLayer.
        """
        if MessagePassing is object:
            raise ImportError("torch_geometric is required for MessagePassingLayer")

        super().__init__(aggr=aggr)

        # Edge update model (phi_e): (h_i, h_j, e_ij) -> e'_ij
        self.edge_model = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, norm),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim, bias=bias),
            Normalization(edge_dim, norm),
        )

        # Node update model (phi_v): (h_i, m_i) -> h'_i
        # m_i is aggregated message
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, norm),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim, bias=bias),
            Normalization(node_dim, norm),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Union[Tuple[torch.Tensor, torch.Tensor], Any]:
        """
        Forward pass aligned with MessagePassing.forward signature.
        """
        # Extract specific tensors from kwargs or args
        x = kwargs.get("x", args[0] if len(args) > 0 else None)
        edge_index = kwargs.get("edge_index", args[1] if len(args) > 1 else None)
        edge_attr = kwargs.get("edge_attr", args[2] if len(args) > 2 else None)

        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("Forward requires x, edge_index, and edge_attr")

        row, col = edge_index

        # 1. Update edges
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_updated = self.edge_model(edge_input)

        # 2. Update nodes
        out_nodes = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)

        return out_nodes, edge_attr_updated

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Create messages. PyG maps 'x_i', 'x_j', etc., based on this signature.
        """
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Update node features using aggregated messages and original features.
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_model(node_input)
