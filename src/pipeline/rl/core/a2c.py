"""
Advantage Actor-Critic (A2C) implementation.

A2C uses a critic network to estimate value function and reduces
variance in policy gradient estimation.

Reference: RL4CO (https://github.com/ai4co/rl4co)
"""

from __future__ import annotations

from typing import Any, Optional, cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.interfaces.env import IEnv
from logic.src.models import CriticNetwork as LegacyCriticNetwork
from logic.src.models.common.autoregressive.constructive import ConstructivePolicy
from logic.src.pipeline.rl.common.base import RL4COLitModule


class A2C(RL4COLitModule):
    """
    Advantage Actor-Critic (A2C) algorithm.

    Uses a learned critic to estimate state values for variance reduction
    in policy gradient estimation. Supports separate optimizer configurations
    for actor and critic.

    Args:
        env: RL environment.
        policy: Actor policy network.
        critic: Critic network for value estimation.
        actor_optimizer: Optimizer for actor ('adam', 'adamw').
        actor_lr: Learning rate for actor.
        critic_optimizer: Optimizer for critic.
        critic_lr: Learning rate for critic.
        entropy_coef: Coefficient for entropy bonus.
        value_loss_coef: Coefficient for value loss.
        normalize_advantage: Whether to normalize advantages.
        **kwargs: Additional arguments for base class.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: Optional[nn.Module] = None,
        actor_optimizer: str = "adam",
        actor_lr: float = 1e-4,
        critic_optimizer: str = "adam",
        critic_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        normalize_advantage: bool = True,
        **kwargs,
    ):
        """
        Initialize A2C algorithm.

        Args:
            env: RL environment.
            policy: Actor policy network.
            critic: Critic policy network.
            actor_optimizer: Optimizer name for actor.
            actor_lr: Learning rate for actor.
            critic_optimizer: Optimizer name for critic.
            critic_lr: Learning rate for critic.
            entropy_coef: Entropy regularization coefficient.
            value_loss_coef: Critic loss coefficient.
            normalize_advantage: Whether to normalize advantages.
            **kwargs: Additional args passed to RL4COLitModule.
        """
        # A2C uses critic baseline
        kwargs["baseline"] = "critic"

        # Cast policy to expected type
        policy_cast = cast(ConstructivePolicy, policy)

        super().__init__(
            env=cast(IEnv, env),
            policy=policy_cast,  # type: ignore[arg-type]
            optimizer=actor_optimizer,  # Ignored as we override configure_optimizers
            optimizer_kwargs={"lr": actor_lr},
            **kwargs,
        )
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["env", "policy", "critic", "kwargs", "generator"])

        # Critic network
        if critic is None:
            critic = LegacyCriticNetwork(
                env_name=env.name if hasattr(env, "name") else "vrpp",
                embed_dim=getattr(policy, "embed_dim", 128),
                hidden_dim=256,
                n_layers=3,
                normalization="batch",
            )
        self.critic = critic

        # Optimizer configs
        self.actor_optimizer_name = actor_optimizer
        self.actor_lr = actor_lr
        self.critic_optimizer_name = critic_optimizer
        self.critic_lr = critic_lr

        # Loss coefficients
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.normalize_advantage = normalize_advantage

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[IEnv] = None,
    ) -> torch.Tensor:
        """
        Calculate A2C loss.

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary.
            batch_idx: Current batch index.
            env: Environment (optional).

        Returns:
            Combined actor + critic loss.
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Get value estimate from critic
        value = self.critic(td).squeeze(-1)

        # Compute advantage
        advantage = reward - value.detach()

        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Actor loss (policy gradient)
        actor_loss = -(advantage * log_likelihood).mean()

        # Critic loss (value function)
        critic_loss = nn.functional.mse_loss(value, reward)

        # Entropy bonus
        entropy = out.get("entropy", torch.tensor(0.0, device=reward.device))
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.mean()

        # Total loss
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/actor_loss", actor_loss, prog_bar=False)
        self.log("train/critic_loss", critic_loss, prog_bar=False)
        self.log("train/entropy", entropy, prog_bar=False)
        self.log("train/advantage_mean", advantage.mean(), prog_bar=False)
        self.log("train/log_likelihood", log_likelihood.mean(), prog_bar=False)
        self.log("train/value", value.mean(), prog_bar=False)

        return total_loss

    def configure_optimizers(self):
        """Configure separate optimizers for actor and critic."""
        # Actor optimizer
        if self.actor_optimizer_name.lower() == "adam":
            actor_opt = torch.optim.Adam(self.policy.parameters(), lr=self.actor_lr)  # type: ignore[attr-defined]
        elif self.actor_optimizer_name.lower() == "adamw":
            actor_opt = torch.optim.AdamW(self.policy.parameters(), lr=self.actor_lr)  # type: ignore[assignment, attr-defined]
        else:
            raise ValueError(f"Unknown actor optimizer: {self.actor_optimizer_name}")

        # Critic optimizer
        if self.critic_optimizer_name.lower() == "adam":
            critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.critic_optimizer_name.lower() == "adamw":
            critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown critic optimizer: {self.critic_optimizer_name}")

        return [actor_opt, critic_opt]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step with both optimizers.
        """
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            actor_opt, critic_opt = optimizers
        else:
            # Fallback if only one optimizer is returned
            actor_opt = optimizers
            critic_opt = None

        # Execute step
        out = self.shared_step(batch, batch_idx, phase="train")
        loss = out["loss"]

        # Manual optimization
        actor_opt.zero_grad()
        if critic_opt is not None:
            critic_opt.zero_grad()

        self.manual_backward(loss)

        actor_opt.step()
        if critic_opt is not None:
            critic_opt.step()

        return loss


__all__ = ["A2C"]
