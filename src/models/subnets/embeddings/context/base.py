"""Abstract base class for context embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ContextEmbedder(nn.Module, ABC):
    """
    Abstract base class for problem-specific context embeddings.
    Responsible for initializing node embeddings and determining step context dimensions.
    """

    def __init__(self, embed_dim: int, node_dim: int, temporal_horizon: int):
        """
        Initialize the ContextEmbedder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon
        self.init_embed: nn.Module | None = None

    @abstractmethod
    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        """Initialize node embeddings from input data."""
        pass

    @property
    @abstractmethod
    def step_context_dim(self) -> int:
        """Get the dimension of the step context."""
        pass

    def forward(self, nodes_or_embeddings: torch.Tensor | dict[str, Any], state: Any | None = None) -> torch.Tensor:
        """
        Forward pass handling both initialization and step context.

        Args:
            nodes_or_embeddings: Raw node features (init) or node embeddings (step).
            state: Optional state object for step context.

        Returns:
            torch.Tensor: Initial node embeddings or step context embedding.
        """
        if state is None:
            return self.init_node_embeddings(nodes_or_embeddings)  # type: ignore[arg-type]
        return self._step_context(nodes_or_embeddings, state)  # type: ignore[arg-type]

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """
        Default step context: project current node embedding.
        Can be overridden by subclasses for problem-specific features.
        """
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")

        batch_size = embeddings.size(0)

        # Ensure current_node is (batch)
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)

        # Gather current node embedding: [batch, 1, embed_dim]
        step_context = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        )

        return step_context
