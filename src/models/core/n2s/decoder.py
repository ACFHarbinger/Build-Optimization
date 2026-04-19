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
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class N2SDecoder(ImprovementDecoder):
    """
    N2S Decoder: Predicts moves within node neighborhoods.
    """

    def __init__(self, embed_dim: int = 128, seed: int = 42, **kwargs):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            seed (int): Description of seed.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(embed_dim=embed_dim)
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        self.seed = seed
        self.init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=self.init_device).manual_seed(self.seed)

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
        Predict move (idx1, idx2).
        """
        h = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        bs, n, _ = h.shape

        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # Mask invalid moves
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores.masked_fill_(mask, float("-inf"))

        logits = scores.view(bs, -1)

        strategy = kwargs.get("strategy", "greedy")
        if strategy == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        # Convert flat index back to (i, j)
        idx1 = torch.div(action_indices, n, rounding_mode="floor")
        idx2 = action_indices % n
        actions = torch.stack([idx1, idx2], dim=-1)

        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions
