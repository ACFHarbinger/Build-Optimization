"""
Neural Neighborhood Search (N2S) Model.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import N2SPolicy


class N2S(nn.Module):
    """
    N2S Model: Neural Neighborhood Search for combinatorial optimization.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[N2SPolicy] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **policy_kwargs,
    ):
        """Initialize Class.

        Args:
            env (Optional[RL4COEnvBase]): Description of env.
            policy (Optional[N2SPolicy]): Description of policy.
            embed_dim (int): Description of embed_dim.
            num_heads (int): Description of num_heads.
            k_neighbors (int): Description of k_neighbors.
            policy_kwargs (Any): Description of policy_kwargs.
        """
        super().__init__()
        if policy is None:
            policy = N2SPolicy(
                embed_dim=embed_dim,
                num_heads=num_heads,
                k_neighbors=k_neighbors,
                **policy_kwargs,
            )
        self.policy = policy
        self.env = env

    def forward(
        self,
        td: Any,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        **kwargs,
    ):
        """Forward pass of the model."""
        env = env or self.env
        return self.policy(td, env, strategy=strategy, **kwargs)
