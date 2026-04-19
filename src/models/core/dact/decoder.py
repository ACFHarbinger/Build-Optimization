"""decoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import decoder
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.policy import ImprovementDecoder


class DACTDecoder(ImprovementDecoder):
    """
    DACT Decoder: Predicts 2-opt moves using cross-attention.

    A 2-opt move is defined by two indices (i, j) in the solution.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, seed: int = 42, **kwargs):
        """Initialize DACTDecoder."""
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.seed = seed
        self.init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=self.init_device).manual_seed(self.seed)

        # Query/Key/Value projections for selecting i and j
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)

        # Score normalization
        self.scale = 1.0 / (embed_dim // num_heads) ** 0.5

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    def __getstate__(self):
        """Prepare state for pickling (handle non-picklable Generator)."""
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def forward(
        self,
        td: TensorDict,
        embeddings: torch.Tensor | Tuple[torch.Tensor, ...],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict two nodes for 2-opt move.

        Returns:
            Tuple of (log_p, actions) where actions is [batch, 2].
        """
        h = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        bs, n, _ = h.shape

        # 1. Project to queries and keys
        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # 3. Mask invalid moves (i >= j for simplicity, or restricted moves)
        # For 2-opt in DACT, we typically pick i in [0, n-1] and j in [0, n-1]
        # But we must have i != j
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Flatten for softmax
        logits = scores.view(bs, -1)  # [bs, n*n]

        # 4. Sample action
        strategy = kwargs.get("strategy", "greedy")
        if strategy == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        # Convert flattened index back to (i, j)
        idx_i = action_indices // n
        idx_j = action_indices % n

        actions = torch.stack([idx_i, idx_j], dim=-1)

        # Log likelihood
        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions
