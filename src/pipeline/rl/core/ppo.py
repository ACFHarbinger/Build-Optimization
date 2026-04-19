"""
PPO algorithm implementation.

Reference:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms.
    arXiv preprint arXiv:1707.06347.
"""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.data.datasets import FastTdDataset
from logic.src.interfaces import ITraversable
from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.utils.data.rl_utils import safe_td_copy


class PPO(RL4COLitModule):
    """
    Proximal Policy Optimization (PPO).

    Implements PPO for constructive routing problems.
    Uses a Critic network as a value baseline and performs multiple epochs
    of optimization per batch of rollouts.

    Reference:
        Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
        Proximal Policy Optimization Algorithms. arXiv:1707.06347
        https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        critic: nn.Module,
        ppo_epochs: int = 10,
        eps_clip: float = 0.2,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 0.5,
        mini_batch_size: int | float = 0.25,
        **kwargs,
    ):
        """
        Initialize PPO module.

        Args:
            critic: Critic network for value estimation.
            ppo_epochs: Number of PPO optimization epochs per batch.
            eps_clip: Clipping parameter for PPO surrogate objective.
            value_loss_weight: Weight for value function loss.
            entropy_weight: Weight for entropy bonus.
            max_grad_norm: Maximum gradient norm for clipping.
            mini_batch_size: Mini-batch size (int or fraction of batch).
            **kwargs: Arguments passed to RL4COLitModule.
        """
        super().__init__(**kwargs)
        self.critic = critic
        self.ppo_epochs = int(ppo_epochs)
        self.eps_clip = float(eps_clip)
        self.value_loss_weight = float(value_loss_weight)
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size

        # We use manual optimization to allow multiple epochs per batch
        self.automatic_optimization = False

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Dummy implementation to satisfy abstract requirement.
        PPO uses manual optimization in training_step.
        """
        return torch.tensor(0.0, device=td.device)

    def training_step(self, batch: TensorDict, batch_idx: int):
        """
        Execute PPO training step with multiple optimization epochs.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.

        Returns:
            Loss tensor from last optimization step.
        """
        if hasattr(self.baseline, "unwrap_batch"):
            batch, _ = self.baseline.unwrap_batch(batch)

        # Ensure batch is a TensorDict (converting from dict if necessary)
        # This handles cases where collation returns a dict or unwrap returns a dict
        if isinstance(batch, ITraversable):
            # If it's a dict, convert to TensorDict
            # We assume it has a batch size equal to the length of its values
            bs = len(next(iter(batch.values())))
            batch = TensorDict(batch, batch_size=[bs])

        # Move to device if needed (usually handled by Lightning but explicit safety)
        if hasattr(batch, "to"):
            batch = batch.to(self.device)

        # DEBUG: Check batch contents
        # print(f"DEBUG: PPO batch keys: {batch.keys()}")
        # if "done" in batch.keys():
        #     print(f"DEBUG: batch['done'].shape: {batch['done'].shape}")
        # print(f"DEBUG: env.batch_size: {self.env.batch_size}")

        # Remove done if present to avoid shape mismatch in reset
        if "done" in batch.keys():
            del batch["done"]

        env = self.env
        td = env.reset(safe_td_copy(batch))

        with torch.no_grad():
            # Sampling rollout
            out = self.policy(td, env, strategy="sampling")
            old_actions = out["actions"]  # (batch, steps)
            old_log_p = out["log_likelihood"]  # (batch)
            old_reward = out["reward"]  # (batch)

            # Initial state value
            self.critic(batch).squeeze(-1)  # (batch)

        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]

        if not isinstance(opt, LightningOptimizer):
            raise TypeError(f"Expected LightningOptimizer, got {type(opt)}")

        # Create DataLoader for mini-batching
        # Add necessary keys to TensorDict
        td.set("logprobs", old_log_p)
        td.set("reward", old_reward)
        td.set("action", old_actions)

        # Determine mini_batch_size
        bs = td.batch_size[0]
        mbs = self.mini_batch_size
        if isinstance(mbs, float):
            mbs = max(1, int(bs * mbs))

        mbs = min(mbs, bs)

        dataset = FastTdDataset(td)
        dataloader = DataLoader(dataset, batch_size=mbs, shuffle=True, collate_fn=FastTdDataset.collate_fn)

        # Initialize metrics for logging
        loss = torch.tensor(0.0, device=td.device)
        actor_loss = torch.tensor(0.0, device=td.device)
        critic_loss = torch.tensor(0.0, device=td.device)
        advantage = torch.tensor(0.0, device=td.device)

        for _ in range(self.ppo_epochs):
            for sub_td in dataloader:
                sub_td = sub_td.to(td.device)
                previous_reward = sub_td["reward"]  # [batch]

                # Re-evaluate logic
                # Need to clone to avoid in-place issues
                new_out = self.policy(safe_td_copy(sub_td), env, actions=sub_td["action"])
                new_log_p = new_out["log_likelihood"]

                # Re-evaluate values
                new_values = self.critic(sub_td).squeeze(-1)  # [batch]

                # Advantage estimation
                advantage = self.calculate_advantages(previous_reward, new_values)

                # Ratio
                # sub_td["logprobs"] is old_log_p
                ratio = self.calculate_ratio(new_log_p, sub_td["logprobs"])

                # Actor Loss
                actor_loss = self.calculate_actor_loss(ratio, advantage)

                # Critic Loss
                critic_loss = self.calculate_critic_loss(new_values, previous_reward)

                # Total Loss
                loss = actor_loss + self.value_loss_weight * critic_loss

                # Entropy
                if self.entropy_weight > 0 and "entropy" in new_out:
                    loss = loss - self.entropy_weight * new_out["entropy"].mean()

                # Manual Optimization Step
                opt.zero_grad()
                self.manual_backward(loss)

                if self.max_grad_norm > 0:
                    torch_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
                    # mypy ignore because we know it's a valid optimizer but the union type is tricky
                    self.clip_gradients(torch_opt, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")  # type: ignore

                opt.step()

        # Log metrics (using last batch approximation)
        self.log("train/reward", old_reward.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/actor_loss", actor_loss)
        self.log("train/critic_loss", critic_loss)
        self.log("train/advantage", advantage.mean())

        # PPO diagnostics (from last mini-batch)
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
            approx_kl = (sub_td["logprobs"] - new_log_p).mean()
        self.log("train/clip_fraction", clip_fraction)
        self.log("train/approx_kl", approx_kl)
        self.log("train/ratio", ratio.mean())
        if self.entropy_weight > 0 and "entropy" in new_out:
            self.log("train/entropy", new_out["entropy"].mean())

        return loss

    def calculate_advantages(self, rewards, values):
        """Estimate advantages (R - V)."""
        advantage = rewards - values.detach()
        if advantage.size(0) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage

    def calculate_ratio(self, new_log_p, old_log_p):
        """Calculate importance ratio."""
        return torch.exp(new_log_p - old_log_p.detach())

    def calculate_actor_loss(self, ratio, advantage):
        """Clipped surrogate objective."""
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
        return -torch.min(surr1, surr2).mean()

    def calculate_critic_loss(self, values, rewards):
        """MSE loss for value function."""
        return nn.MSELoss()(values, rewards)

    def configure_optimizers(self):
        """
        Configure optimizer for policy and critic parameters.

        Returns:
            Adam optimizer with combined policy and critic parameters.
        """
        # Combined parameters from policy and critic
        params = list(self.policy.parameters()) + list(self.critic.parameters())  # type: ignore[attr-defined]
        return torch.optim.Adam(params, **self.optimizer_kwargs)
