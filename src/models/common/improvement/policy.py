"""improvement_policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import improvement_policy
"""

from __future__ import annotations

import random
from abc import ABC
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs import get_env
from logic.src.envs.base.base import RL4COEnvBase

from .decoder import ImprovementDecoder
from .encoder import ImprovementEncoder


class ImprovementPolicy(nn.Module, ABC):
    """
    Base class for improvement policies.

    Improvement policies take an instance + a solution as input and output a specific
    operator that changes the current solution to a new one.
    """

    def __init__(
        self,
        encoder: Optional[ImprovementEncoder] = None,
        decoder: Optional[ImprovementDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize ImprovementPolicy."""
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

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass of the policy using an iterative improvement loop.

        Args:
            td: TensorDict containing the environment state and current solution.
            env: Environment to use for decoding.
            strategy: Decoding strategy (greedy, sampling, etc.).
            num_starts: Number of solution starts.
            max_steps: Maximum number of improvement steps.
            phase: Phase of the algorithm (train, val, test).
            return_actions: Whether to return the actions.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally actions.
        """
        if env is None:
            # Try to get from name or default to base
            env = get_env(self.env_name or "tsp_kopt")

        # Initial solution generation (done by env.reset)
        from logic.src.utils.decoding import batchify, unbatchify

        td = env.reset(td)

        # Batch for multiple starts if requested
        if num_starts > 1:
            td = batchify(td, num_starts)

        # Iterative improvement loop
        log_probs = []
        actions = []

        # Default steps from td or config
        if max_steps is None:
            max_steps_td = td.get("max_steps", None)
            max_steps = int(max_steps_td.item()) if max_steps_td is not None else 10

        assert isinstance(max_steps, int), f"max_steps must be an int, got {type(max_steps)}"

        for _i in range(max_steps):
            # 1. Encode current state
            if self.encoder is None:
                raise ValueError("Encoder must be provided for ImprovementPolicy")
            embeddings = self.encoder(td)

            # 2. Predict move via decoder
            if self.decoder is None:
                raise ValueError("Decoder must be provided for ImprovementPolicy")
            log_p, move = self.decoder(td, embeddings, env, strategy=strategy, **kwargs)

            # 3. Apply move
            td.set("action", move)
            td = env.step(td)["next"]

            log_probs.append(log_p)
            actions.append(move)

            if td["done"].all():
                break

        # Collect results
        out = {
            "reward": env.get_reward(td),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
        }

        if return_actions:
            out["actions"] = torch.stack(actions, dim=1)

        # Unbatch if multiple starts
        if num_starts > 1:
            out["reward"] = unbatchify(out["reward"], num_starts)
            out["log_likelihood"] = unbatchify(out["log_likelihood"], num_starts)
            if return_actions:
                out["actions"] = unbatchify(out["actions"], num_starts)

        return out
