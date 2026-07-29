"""VRPP specific context embedding."""

from __future__ import annotations

from typing import Any

import torch

from .env import EnvState


class VRPPState(EnvState):
    """Context embedding for VRPP."""

    def __init__(self, embed_dim: int):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
        """
        super().__init__(embed_dim, step_context_dim=1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """state embedding.

        Args:
            embeddings (torch.Tensor): Description of embeddings.
            td (Any): Description of td.

        Returns:
            Any: Description of return value.
        """
        # Remaining length or capacity
        # We use 'max_length' or 'remaining_capacity'
        if "remaining_length" in td.keys():
            state = td["remaining_length"]
        elif "remaining_capacity" in td.keys():
            state = td["remaining_capacity"]
        else:
            state = torch.ones(embeddings.size(0), device=embeddings.device)

        if state.dim() == 1:
            state = state[:, None]

        return state
