"""
NARGNN Model: REINFORCE wrapper for non-autoregressive GNNs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import NARGNNPolicy


class NARGNN(nn.Module):
    """
    NARGNN Model.

    REINFORCE-based training wrapper for NARGNNPolicy.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        baseline: Optional[str] = "rollout",
        **kwargs,
    ):
        """
        Initialize NARGNN model.
        """
        super().__init__()
        self.policy = NARGNNPolicy(
            embed_dim=embed_dim,
            env_name=env_name,
            num_layers_heatmap_generator=num_layers_heatmap_generator,
            num_layers_graph_encoder=num_layers_graph_encoder,
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
