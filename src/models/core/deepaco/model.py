"""
DeepACO Model: REINFORCE wrapper for training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import DeepACOPolicy


class DeepACO(nn.Module):
    """
    DeepACO Model.

    REINFORCE-based training wrapper for DeepACOPolicy.
    Supports baseline subtraction for variance reduction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        baseline: str = "rollout",
        env_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize DeepACO model.

        Args:
            embed_dim: Embedding dimension.
            num_encoder_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            n_ants: Number of ants.
            n_iterations: ACO iterations.
            alpha: Pheromone weight.
            beta: Heuristic weight.
            rho: Evaporation rate.
            use_local_search: Apply 2-opt.
            baseline: Baseline type ("rollout", "exponential", "none").
            env_name: Environment name.
        """
        super().__init__()
        self.policy = DeepACOPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_local_search=use_local_search,
            env_name=env_name,
            **kwargs,
        )
        self.baseline_type = baseline
        self._baseline_val = None

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            td: TensorDict with problem instance.
            env: Environment.

        Returns:
            Dictionary with loss, reward, etc.
        """
        out = self.policy(td, env, **kwargs)

        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Compute baseline
        if self.baseline_type == "exponential":
            if self._baseline_val is None:
                self._baseline_val = reward.mean().detach()
            else:
                self._baseline_val = 0.8 * self._baseline_val + 0.2 * reward.mean().detach()
            baseline = self._baseline_val
        elif self.baseline_type == "rollout":
            # Use batch mean as baseline
            baseline = reward.mean()
        else:
            baseline = 0.0

        # REINFORCE loss
        advantage = reward - baseline
        loss = -(advantage.detach() * log_likelihood).mean()

        out["loss"] = loss
        out["baseline"] = baseline
        return out

    def set_strategy(self, strategy: str, **kwargs):
        """Set strategy for evaluation."""
        self.policy.set_strategy(strategy, **kwargs)

    def eval(self):
        """Set to evaluation mode."""
        super().eval()
        return self
