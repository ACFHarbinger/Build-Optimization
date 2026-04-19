"""
Base policy classes for constructive and improvement methods.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class ConstructivePolicy(nn.Module, ABC):
    """
    Base class for constructive (autoregressive) policies.

    Constructive policies build solutions step-by-step by selecting
    one node at a time until the solution is complete.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize ConstructivePolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.seed = seed
        self.device = torch.device(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.rng = random.Random(seed) if seed is not None else random.Random()

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

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Full forward pass: encode + decode until done.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions.
            strategy: Decoding strategy ("sampling", "greedy", "beam_search").
            num_starts: Number of solution starts for multi-start methods.

        Returns:
            Dictionary containing:
                - reward: Final reward/cost
                - log_likelihood: Log probability of solution
                - tour: Sequence of visited nodes
        """
        raise NotImplementedError

    def _select_action(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        strategy: str = "sampling",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action based on logits and decode type using DecodingStrategy.

        Args:
            logits: Action logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]
            strategy: Decoding strategy name
            **kwargs: Additional arguments for decoding strategy (temperature, etc.)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        # Get strategy (can be cached if needed)
        from logic.src.utils.decoding import get_decoding_strategy

        decoder_strategy = get_decoding_strategy(strategy, **kwargs)

        # Step
        result = decoder_strategy.step(logits, mask)
        if len(result) == 3:
            action, log_prob, entropy = result
        else:
            action, log_prob = result[:2]  # type: ignore[misc]
            entropy = result[2] if len(result) > 2 else torch.tensor(0.0)
        return action, log_prob, entropy
