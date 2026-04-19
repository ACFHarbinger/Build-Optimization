"""generic.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import generic
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from logic.src.constants.models import NODE_DIM

from .base import ContextEmbedder


class GenericContextEmbedder(ContextEmbedder):
    """
    Generic context embedder for problems without specific implementations.
    Embeds depot (2D) and nodes (node_dim) simply.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            node_dim (int): Description of node_dim.
            temporal_horizon (int): Description of temporal_horizon.
        """
        super().__init__(embed_dim, node_dim, temporal_horizon)
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

        # Step context projection: usually just current node for generic
        # If step_context_dim == embed_dim, it's just the gathered embedding
        self.project_step_context = (
            nn.Identity() if self.step_context_dim == embed_dim else nn.Linear(self.step_context_dim, embed_dim)
        )

    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        """Init node embeddings.

        Args:
            nodes (dict[str, Any]): Description of nodes.

        Returns:
            Any: Description of return value.
        """
        # Fallback to 'loc' or 'locs'
        locs_key = "locs" if "locs" in nodes else "loc"
        node_features = nodes[locs_key]

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),  # type: ignore[misc]
            ),
            1,
        )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """
        Default step context: return gathered current node embedding.
        """
        batch_size = embeddings.size(0)
        current_node = state.get_current_node()
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)

        # Gather current node embedding: [batch, 1, embed_dim]
        step_context = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        )

        return self.project_step_context(step_context)

    @property
    def step_context_dim(self) -> int:
        """
        Generic context just returns the current node embedding.
        """
        return self.embed_dim
