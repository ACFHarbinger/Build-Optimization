"""env.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import env
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EnvState(nn.Module):
    """Base class for environment context embeddings."""

    def __init__(self, embed_dim: int, step_context_dim: int = 0, node_dim: int = 0):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            step_context_dim (int): Description of step_context_dim.
            node_dim (int): Description of node_dim.
        """
        super().__init__()
        self.embed_dim = embed_dim
        if step_context_dim > 0:
            self.project_context = nn.Linear(step_context_dim, embed_dim)
        else:
            self.project_context: nn.Module | None = None  # type: ignore[assignment, no-redef]

    def forward(
        self,
        embeddings: torch.Tensor,
        td: Any,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, num_nodes, embed_dim]
            td: TensorDict containing the current state
        """
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)

        if self.project_context is not None:
            return cur_node_embedding + self.project_context(state_embedding)

        return cur_node_embedding + state_embedding

    def _cur_node_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """cur node embedding.

        Args:
            embeddings (torch.Tensor): Description of embeddings.
            td (Any): Description of td.

        Returns:
            Any: Description of return value.
        """
        # Current node is usually the last visited node
        cur_node = td["current_node"]
        if cur_node.dim() == 1:
            cur_node = cur_node[:, None]
        return torch.gather(embeddings, 1, cur_node.expand(-1, embeddings.size(-1))[:, None, :]).squeeze(1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """state embedding.

        Args:
            embeddings (torch.Tensor): Description of embeddings.
            td (Any): Description of td.

        Returns:
            Any: Description of return value.
        """
        return torch.zeros(embeddings.size(0), self.embed_dim, device=embeddings.device)
