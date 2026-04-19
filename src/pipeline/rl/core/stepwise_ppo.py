"""
Stepwise PPO implementation.

This algorithm decomposes the solution construction/improvement into
individual steps and provides reward signals for each transition.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.data.datasets import FastTdDataset
from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.utils.data.rl_utils import safe_td_copy


class StepwisePPO(RL4COLitModule):
    """
    Stepwise Proximal Policy Optimization (StepwisePPO).

    Decomposes the sequence generation into individual MDP steps.
    Each node selection or local search move is treated as a transition
    with its own reward (e.g., incremental distance reduction).
    """

    def __init__(
        self,
        critic: nn.Module,
        ppo_epochs: int = 10,
        eps_clip: float = 0.2,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.01,
        max_grad_norm: float = 0.5,
        mini_batch_size: int | float = 0.25,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        **kwargs,
    ):
        """
        Initialize StepwisePPO.

        Args:
            critic: Critic network for value estimation at each step.
            ppo_epochs: Number of optimization epochs per rollout.
            eps_clip: PPO clipping parameter.
            value_loss_weight: Weight for value function loss.
            entropy_weight: Weight for entropy bonus.
            max_grad_norm: Gradient clipping norm.
            mini_batch_size: Batch size for PPO updates.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            **kwargs: Arguments for RL4COLitModule.
        """
        super().__init__(**kwargs)
        self.critic = critic
        self.ppo_epochs = int(ppo_epochs)
        self.eps_clip = float(eps_clip)
        self.value_loss_weight = float(value_loss_weight)
        self.entropy_weight = float(entropy_weight)
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Manual optimization for multiple epochs
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """Execute one StepwisePPO training step."""
        env = self.env
        td = env.reset(batch)

        # 1. Rollout Trajectories
        # We need to collect (s, a, log_p, reward, next_s, done)
        experiences = []

        # Initial value
        v_current = self.critic(td).squeeze(-1)

        # Loop until all batches are done
        while not td["done"].all():
            # Current state (detached for storage)
            td_current = safe_td_copy(td)

            # Policy forward (one step)
            # Note: For AttentionModel, we need to ensure it can do one step.
            # Most of our policies currently do the whole loop.
            # We'll use the policy's decoder directly if available.
            if hasattr(self.policy, "pre_step_encode"):
                embeddings = self.policy.pre_step_encode(td)
            else:
                # Fallback: assume constructive or improvement policy has encoder
                assert self.policy.encoder is not None, "Policy must have an encoder"
                embeddings = self.policy.encoder(td)

            # One step of decoding
            assert self.policy.decoder is not None, "Policy must have a decoder"
            log_p, action = self.policy.decoder(td, embeddings, env, strategy="sampling", return_pi=True)

            # Step environment
            td["action"] = action
            td_next = env.step(td)["next"]

            # Step reward: reduction in total cost or immediate reward
            # If env returns cumulative reward, immediate = reward_next - reward_prev
            # But most of our envs return total distance at each step.
            # We want the reduction in cost.
            r_immediate = td_next["reward"] - td.get("reward", torch.zeros_like(td_next["reward"]))

            # Store experience
            experiences.append(
                {
                    "td": td_current,
                    "action": action,
                    "log_p": log_p,
                    "reward": r_immediate.squeeze(-1),
                    "value": v_current,
                    "done": td_next["done"].squeeze(-1),
                }
            )

            td = td_next
            v_current = self.critic(td).squeeze(-1)

        # 2. Process Trajectories (GAE)
        # Flatten experiences into a single TensorDict
        data = self._process_experiences(experiences)

        # 3. PPO Optimization Loop
        dataloader = DataLoader(
            FastTdDataset(data), batch_size=self._get_mbs(data), shuffle=True, collate_fn=FastTdDataset.collate_fn
        )
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]

        sub_td = None
        new_log_p = None
        loss = torch.tensor(0.0, device=self.device)
        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)
        ratio = torch.tensor(0.0, device=self.device)
        for _ in range(self.ppo_epochs):
            for sub_td in dataloader:
                sub_td = sub_td.to(self.device)

                # Re-evaluate logic (must match the step above)
                # For simplicity, we assume we can run the one-step decoder
                assert self.policy.encoder is not None
                emb = self.policy.encoder(sub_td)
                assert self.policy.decoder is not None
                new_log_p, _ = self.policy.decoder(sub_td, emb, env, actions=sub_td["action"])
                new_values = self.critic(sub_td).squeeze(-1)

                # Ratio
                ratio = torch.exp(new_log_p - sub_td["log_p"])

                # Actor Loss
                surr1 = ratio * sub_td["advantage"]
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * sub_td["advantage"]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss
                critic_loss = nn.functional.mse_loss(new_values, sub_td["return"])

                # Total Loss
                loss = actor_loss + self.value_loss_weight * critic_loss

                # Optimize
                opt.zero_grad()
                self.manual_backward(loss)
                if self.max_grad_norm > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")  # type: ignore[arg-type]
                opt.step()

        # Logging
        self.log("train/avg_reward", data["reward"].mean())
        self.log("train/loss", loss)
        self.log("train/actor_loss", actor_loss)
        self.log("train/critic_loss", critic_loss)
        self.log("train/advantage", data["advantage"].mean())
        # StepwisePPO diagnostics (from last mini-batch)
        if sub_td is not None and new_log_p is not None:
            with torch.no_grad():
                clip_fraction = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
                approx_kl = (sub_td["log_p"] - new_log_p).mean()
            self.log("train/clip_fraction", clip_fraction)
            self.log("train/approx_kl", approx_kl)
        return loss

    def _process_experiences(self, experiences: list) -> TensorDict:
        """Compute GAE advantages and returns."""
        rewards = torch.stack([e["reward"] for e in experiences], dim=1)  # [B, T]
        values = torch.stack([e["value"] for e in experiences], dim=1)  # [B, T]
        dones = torch.stack([e["done"] for e in experiences], dim=1)  # [B, T]

        bs, T = rewards.shape
        next_values = torch.zeros((bs, T), device=rewards.device)
        next_values[:, :-1] = values[:, 1:]

        # GAE Calculation
        advantages = torch.zeros_like(rewards)

        # FIX: Initialize as a Tensor (or 0.0) to avoid int vs Tensor conflicts
        last_gae = torch.zeros(bs, device=rewards.device)

        for t in reversed(range(T)):
            mask = 1.0 - dones[:, t].float()
            delta = rewards[:, t] + self.gamma * next_values[:, t] * mask - values[:, t]
            # Now the assignment is consistent: Tensor = Tensor + (scalar * scalar * Tensor * Tensor)
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values
        for t in reversed(range(T)):
            # delta = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
            mask = 1.0 - dones[:, t].float()
            delta = rewards[:, t] + self.gamma * next_values[:, t] * mask - values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae

        returns = advantages + values

        # Flatten batch and time dimensions for mini-batching
        # Flatten batch and time dimensions for mini-batching
        # Stack the individual TensorDicts along dimension 1 (time)
        data = torch.stack([e["td"] for e in experiences], dim=1)  # [B, T, ...]

        # Build flattened TensorDict
        data["action"] = torch.stack([e["action"] for e in experiences], dim=1)  # type: ignore[index]
        data["log_p"] = torch.stack([e["log_p"] for e in experiences], dim=1)  # type: ignore[index]
        data["advantage"] = advantages  # type: ignore[index]
        data["return"] = returns  # type: ignore[index]
        data["reward"] = rewards  # type: ignore[index]  # per-step reward for logging

        # Reshape to [B*T]
        return data.view(bs * T)

    def _get_mbs(self, td):
        """get mbs.

        Args:
            td (Any): Description of td.

        Returns:
            Any: Description of return value.
        """
        bs = td.batch_size[0]
        mbs = self.mini_batch_size
        if isinstance(mbs, float):
            mbs = int(bs * mbs)
        return min(max(1, mbs), bs)

    def calculate_loss(self, td, out, batch_idx, env=None):
        """Calculate loss.

        Args:
            td (Any): Description of td.
            out (Any): Description of out.
            batch_idx (Any): Description of batch_idx.
            env (Any): Description of env.

        Returns:
            Any: Description of return value.
        """
        return torch.tensor(0.0, device=td.device)

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            Any: Description.
        """
        params = list(self.policy.parameters()) + list(self.critic.parameters())  # type: ignore[attr-defined]
        return torch.optim.Adam(params, **self.optimizer_kwargs)
