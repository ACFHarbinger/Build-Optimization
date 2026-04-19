"""VRPP specific context embedder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from logic.src.constants.models import NODE_DIM, VRPP_STEP_CONTEXT_OFFSET

from .base import ContextEmbedder


class VRPPContextEmbedder(ContextEmbedder):
    """Context Embedder for VRP with Profits (VRPP) families."""

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            node_dim (int): Description of node_dim.
            temporal_horizon (int): Description of temporal_horizon.
        """
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim
        if temporal_horizon > 0:
            input_dim += temporal_horizon

        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

        # Step context projection: current_node_embed + state_features
        self.project_step_context = nn.Linear(self.step_context_dim, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        """Init node embeddings.

        Args:
            nodes (dict[str, Any]): Description of nodes.
            temporal_features (bool): Description of temporal_features.

        Returns:
            Any: Description of return value.
        """
        # Initial node embedding: locs + waste
        # Waste is usually [batch_size, num_nodes]
        # locs is [batch_size, num_nodes, 2]

        locs = nodes.get("locs") if "locs" in nodes.keys() else nodes.get("loc")
        if locs is None:
            # Fallback for missing locs
            return torch.zeros(1, 1, self.embed_dim, device=nodes.device if hasattr(nodes, "device") else None)

        # Use waste for node quantities
        waste = nodes.get("waste")
        if waste is None:
            waste = torch.zeros(locs.shape[0], locs.shape[1], 1, device=locs.device) if self.node_dim > 2 else None

        node_features: list[torch.Tensor] = [locs]
        if waste is not None:
            if waste.dim() == 2:
                waste = waste.unsqueeze(-1)
            node_features.append(waste)

        # Add temporal features if available
        if self.temporal_horizon > 0:
            temp_feat = nodes.get("temporal_features")
            if temp_feat is not None:
                node_features.append(temp_feat)
            else:
                # Pad with zeros if missing
                node_features.append(torch.zeros(locs.size(0), locs.size(1), self.temporal_horizon, device=locs.device))

        node_features_tensor = torch.cat(node_features, -1)

        # Determine if locs already contains the depot
        depot = nodes["depot"]
        waste_raw = nodes.get("waste")

        # Check if already concatenated: shape matches OR first element matches depot
        is_concatenated = waste_raw is not None and (
            locs.size(1) == waste_raw.size(1) + 1
            or (locs.size(1) == waste_raw.size(1) and torch.allclose(locs[..., 0, :], depot, atol=1e-4))
        )

        if is_concatenated:
            return self.init_embed(node_features_tensor)  # type: ignore[misc]

        # Traditional: separate depot and customers
        return torch.cat(
            (
                self.init_embed_depot(depot)[:, None, :],
                self.init_embed(node_features_tensor),  # type: ignore[misc]
            ),
            dim=1,
        )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """
        Get VRPP-specific step context.
        Concatenates current node embedding and remaining capacity/load.
        """
        batch_size = embeddings.size(0)
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")

        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)

        # 1. Current node embedding [batch, 1, embed_dim]
        current_node_embed = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        )

        # 2. State features (e.g. current_load) [batch, 1, 1]
        # In TensorDictStateWrapper, we often have 'current_load' or 'remaining_capacity'
        state_features = state.get("current_load", None)
        if state_features is None:
            state_features = state.get("remaining_capacity", torch.zeros(batch_size, 1, device=embeddings.device))

        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(-1)
        if state_features.dim() == 2:
            state_features = state_features.unsqueeze(1)

        # Concat and project
        combined = torch.cat([current_node_embed, state_features], dim=-1)
        return self.project_step_context(combined)

    @property
    def step_context_dim(self) -> int:
        """
        VRPP context usually adds remaining length/capacity info (OFFSET)
        to the current node embedding (embed_dim).
        """
        return self.embed_dim + VRPP_STEP_CONTEXT_OFFSET
