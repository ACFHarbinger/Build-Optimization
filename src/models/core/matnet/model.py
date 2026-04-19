"""
MatNet Model: REINFORCE wrapper for matrix-based models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import MatNetPolicy


class MatNet(nn.Module):
    """
    MatNet Model.

    REINFORCE-based training wrapper for MatNetPolicy.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 5,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "instance",
        baseline: str = "rollout",
        env_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MatNet model.
        """
        super().__init__()
        self.policy = MatNetPolicy(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            normalization=normalization,
            problem=None,  # Handled by policy if needed
            **kwargs,
        )
        self.baseline_type = baseline
        self._baseline_val = None

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.
        """
        out = self.policy(td, env, **kwargs)

        # MatNetPolicy returns {"log_p": log_p, "actions": actions}
        # We need to compute reward if not present
        if "reward" not in out and env is not None:
            out["reward"] = env.get_reward(td, out["actions"])

        if "reward" in out and "log_p" in out:
            reward = out["reward"]
            log_likelihood = out["log_p"]

            # Compute baseline
            if self.baseline_type == "exponential":
                if self._baseline_val is None:
                    self._baseline_val = reward.mean().detach()
                else:
                    self._baseline_val = 0.8 * self._baseline_val + 0.2 * reward.mean().detach()
                baseline = self._baseline_val
            elif self.baseline_type == "rollout":
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
        """Set strategy.

        Args:
            strategy (str): Description of strategy.
            kwargs (Any): Description of kwargs.
        """
        self.policy.set_strategy(strategy, **kwargs)
