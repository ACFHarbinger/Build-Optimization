"""WC specific context embedder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from logic.src.constants.models import NODE_DIM, WC_STEP_CONTEXT_OFFSET

from .base import ContextEmbedder


class WCVRPContextEmbedder(ContextEmbedder):
    """Context Embedder for Waste Collection Vehicle Routing Problems (WCVRP)."""

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
        locs = nodes.get("locs")
        if locs is None:
            locs = nodes.get("loc")
        assert locs is not None, "nodes must contain 'locs' or 'loc'"
        waste = nodes.get("waste")

        if waste is None:
            # Fallback with zeros
            waste = torch.zeros(locs.size(0), locs.size(1), device=locs.device)

        if waste.dim() == 2:
            waste = waste.unsqueeze(-1)

        node_features: list[torch.Tensor] = [locs, waste]

        # Add temporal features if available
        if self.temporal_horizon > 0:
            if temporal_features and "temporal_features" in nodes:
                node_features.append(nodes["temporal_features"])
            else:
                # Pad with zeros if missing
                node_features.append(torch.zeros(locs.size(0), locs.size(1), self.temporal_horizon, device=locs.device))

        node_features = torch.cat(node_features, -1)  # type: ignore[assignment]

        # Determine if locs already contains the depot
        depot = nodes["depot"]
        is_concatenated = False

        # Heuristic 1: Shape check vs waste
        # If locs has matching size with waste, it might be customer-only OR already both-prepended.
        if locs.size(1) == waste.size(1):
            # Both same size. Check values.
            if torch.allclose(locs[..., 0, :], depot, atol=1e-4):
                is_concatenated = True
        elif locs.size(1) == waste.size(1) + 1:
            # locs has one more than waste. Assume it's the depot.
            is_concatenated = True

        if is_concatenated:
            # Already has depot, just embed
            return self.init_embed(node_features)  # type: ignore[misc]
        else:
            # Traditional: separate depot and customers
            return torch.cat(
                (
                    self.init_embed_depot(depot)[:, None, :],
                    self.init_embed(node_features),  # type: ignore[misc]
                ),
                1,
            )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """
        Get WCVRP-specific step context.
        Concatenates current node embedding and state features (capacity, time).
        """
        batch_size = embeddings.size(0)
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")

        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)

        # 1. Current node embedding [batch, 1, embed_dim]
        current_node_embed = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        )

        # 2. State features [batch, 1, WC_STEP_CONTEXT_OFFSET]
        # Common features: remaining_capacity, current_time
        cap = state.get("remaining_capacity", torch.zeros(batch_size, 1, device=embeddings.device))
        time = state.get("current_time", torch.zeros(batch_size, 1, device=embeddings.device))

        if cap.dim() == 1:
            cap = cap.unsqueeze(-1)
        if time.dim() == 1:
            time = time.unsqueeze(-1)

        state_features = torch.cat([cap, time], dim=-1)
        if state_features.dim() == 2:
            state_features = state_features.unsqueeze(1)

        # Concat and project
        combined = torch.cat([current_node_embed, state_features], dim=-1)
        return self.project_step_context(combined)

    @property
    def step_context_dim(self) -> int:
        """
        WCVRP context adds remaining capacity and time (OFFSET)
        to the current node embedding (embed_dim).
        """
        return self.embed_dim + WC_STEP_CONTEXT_OFFSET
