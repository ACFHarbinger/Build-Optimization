"""shared_critic.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import shared_critic
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.critic_network.policy import create_critic_from_actor
from logic.src.utils.functions.rl import ensure_tensordict

from .base import Baseline


class SharedBaseline(Baseline):
    """
    Shared baseline using a critic that shares the encoder with the actor.

    Creates a critic network from the actor's encoder via weight sharing
    (deepcopy). This enables the critic to leverage the same learned
    representations while having its own value head.
    """

    def __init__(
        self,
        critic: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Initialize SharedBaseline.

        Args:
            critic: Pre-built critic module. If None, must call setup() with a policy
                    that has a ``create_critic_from_actor`` compatible encoder.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.critic = critic

    def setup(self, policy: nn.Module):
        """
        Build the critic from the actor's encoder.

        Args:
            policy: Actor policy with an encoder attribute.
        """
        if self.critic is not None:
            return  # Already built

        self.critic = create_critic_from_actor(policy)

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using shared critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (fallback shape).
            env: Environment (unused).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)

        td = ensure_tensordict(td, next(iter(self.critic.parameters())).device)
        return self.critic(td).squeeze(-1)

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the shared critic."""
        return list(self.critic.parameters()) if self.critic is not None else []
