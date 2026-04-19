"""
Group Sequence Policy Optimization (GSPO) algorithm.

Reference:
    Zheng, C., Liu, S., Li, M., et al. (2025). Group Sequence Policy Optimization.
    arXiv preprint arXiv:2507.18071.
    https://arxiv.org/abs/2507.18071
"""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from tensordict import TensorDict
from torch.utils.data import DataLoader

from logic.src.data.datasets import FastTdDataset
from logic.src.interfaces import ITraversable
from logic.src.pipeline.rl.core.ppo import PPO
from logic.src.utils.data.rl_utils import safe_td_copy


class GSPO(PPO):
    """
    Group Sequence Policy Optimization (GSPO).

    GSPO introduces two key modifications to PPO:
    1. Group-based advantage normalization: A_i = (R_i - mean(R_group)) / std(R_group)
    2. Sequence-level importance ratio: r = exp((log π_new - log π_old) / L)
       where L is the sequence length (number of actions in the trajectory)

    The sequence-level normalization makes the importance ratio more stable for
    variable-length sequences common in routing problems.

    Reference:
        Zheng, C., Liu, S., Li, M., et al. (2025). Group Sequence Policy Optimization.
        arXiv:2507.18071. https://arxiv.org/abs/2507.18071
    """

    def __init__(
        self,
        use_sequence_normalization: bool = True,
        **kwargs,
    ):
        """
        Initialize GSPO module.

        Args:
            use_sequence_normalization: If True, normalize importance ratio by sequence length.
            **kwargs: Arguments passed to PPO.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["critic", "env", "policy", "kwargs", "generator"])
        self.use_sequence_normalization = use_sequence_normalization

    def training_step(self, batch: TensorDict, batch_idx: int):  # noqa: ARG002
        """
        Execute GSPO training step with sequence-level importance ratio.

        The key difference from PPO is that we track sequence lengths and use them
        to normalize the importance ratio calculation.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.

        Returns:
            Loss tensor from last optimization step.
        """
        if hasattr(self.baseline, "unwrap_batch"):
            batch, _ = self.baseline.unwrap_batch(batch)

        # Ensure batch is a TensorDict
        if isinstance(batch, ITraversable):
            bs = len(next(iter(batch.values())))
            batch = TensorDict(batch, batch_size=[bs])

        if hasattr(batch, "to"):
            batch = batch.to(self.device)

        # Remove done if present to avoid shape mismatch
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

            # Calculate sequence lengths from actions
            # For routing problems, sequence length is typically the number of steps taken
            if old_actions.dim() == 2:
                # Actions shape: (batch, max_steps)
                # Calculate actual sequence lengths (excluding padding/depot returns)
                seq_lengths = (old_actions != 0).sum(dim=1).float()  # (batch,)
                # Ensure minimum length of 1 to avoid division by zero
                seq_lengths = torch.clamp(seq_lengths, min=1.0)
            else:
                # Fallback: assume uniform sequence length
                seq_lengths = torch.ones(old_log_p.size(0), device=old_log_p.device)

            # Initial state value (not used in GSPO but kept for compatibility)
            self.critic(batch).squeeze(-1)

        # PPO Optimization Loop
        opt = self.optimizers()

        if isinstance(opt, list):
            opt = opt[0]

        if not isinstance(opt, LightningOptimizer):
            raise TypeError(f"Expected LightningOptimizer, got {type(opt)}")

        # Create DataLoader for mini-batching
        td.set("logprobs", old_log_p)
        td.set("reward", old_reward)
        td.set("action", old_actions)
        td.set("seq_lengths", seq_lengths)  # Store sequence lengths for later use

        # Determine mini_batch_size
        bs = td.batch_size[0]
        mbs = self.mini_batch_size
        if isinstance(mbs, float):
            mbs = max(1, int(bs * mbs))
        mbs = min(mbs, bs)

        dataset = FastTdDataset(td)  # type: ignore[arg-type]
        dataloader = DataLoader(dataset, batch_size=mbs, shuffle=True, collate_fn=FastTdDataset.collate_fn)

        # Initialize metrics for logging (will be overwritten in loop)
        loss = torch.tensor(0.0, device=td.device)
        actor_loss = torch.tensor(0.0, device=td.device)
        critic_loss = torch.tensor(0.0, device=td.device)
        advantage = torch.tensor(0.0, device=td.device)
        ratio = torch.tensor(1.0, device=td.device)

        # Track last batch variables for logging
        last_sub_td = None
        last_new_log_p = None
        last_sub_seq_lengths = None
        last_new_out = None

        for _ in range(self.ppo_epochs):
            for sub_td in dataloader:
                sub_td = sub_td.to(td.device)
                previous_reward = sub_td["reward"]
                sub_seq_lengths = sub_td["seq_lengths"]

                # Re-evaluate policy
                new_out = self.policy(safe_td_copy(sub_td), env, actions=sub_td["action"])
                new_log_p = new_out["log_likelihood"]

                # Re-evaluate values
                new_values = self.critic(sub_td).squeeze(-1)

                # Group-based advantage estimation (same as PPO)
                advantage = self.calculate_advantages(previous_reward, new_values)

                # GSPO-specific: Sequence-level importance ratio
                ratio = self.calculate_ratio_gspo(new_log_p, sub_td["logprobs"], sub_seq_lengths)

                # Actor Loss (standard PPO clipped surrogate)
                actor_loss = self.calculate_actor_loss(ratio, advantage)

                # Critic Loss
                critic_loss = self.calculate_critic_loss(new_values, previous_reward)

                # Total Loss
                loss = actor_loss + self.value_loss_weight * critic_loss

                # Entropy bonus
                if self.entropy_weight > 0 and "entropy" in new_out:
                    loss = loss - self.entropy_weight * new_out["entropy"].mean()

                # Manual Optimization Step
                opt.zero_grad()
                self.manual_backward(loss)

                if self.max_grad_norm > 0:
                    torch_opt = opt.optimizer if hasattr(opt, "optimizer") else opt
                    self.clip_gradients(
                        torch_opt,  # type: ignore
                        gradient_clip_val=self.max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )

                opt.step()

                # Store last batch variables for logging
                last_sub_td = sub_td
                last_new_log_p = new_log_p
                last_sub_seq_lengths = sub_seq_lengths
                last_new_out = new_out

        # Log metrics (using last batch approximation)
        self.log("train/reward", old_reward.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/actor_loss", actor_loss)
        self.log("train/critic_loss", critic_loss)
        self.log("train/advantage", advantage.mean())

        # GSPO diagnostics (using last batch values)
        if last_sub_td is not None and last_new_log_p is not None and last_sub_seq_lengths is not None:
            with torch.no_grad():
                clip_fraction = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
                approx_kl = (last_sub_td["logprobs"] - last_new_log_p).mean()
                avg_seq_length = last_sub_seq_lengths.mean()

            self.log("train/clip_fraction", clip_fraction)
            self.log("train/approx_kl", approx_kl)
            self.log("train/ratio", ratio.mean())
            self.log("train/avg_seq_length", avg_seq_length)

        if self.entropy_weight > 0 and last_new_out is not None and "entropy" in last_new_out:
            self.log("train/entropy", last_new_out["entropy"].mean())

        return loss

    def calculate_ratio_gspo(
        self,
        new_log_p: torch.Tensor,
        old_log_p: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate sequence-level importance ratio for GSPO.

        GSPO normalizes the log probability difference by sequence length:
        r = exp((log π_new - log π_old) / L)

        This makes the importance ratio more stable for variable-length sequences.

        Args:
            new_log_p: Log probabilities from current policy (batch,)
            old_log_p: Log probabilities from old policy (batch,)
            seq_lengths: Sequence lengths for each trajectory (batch,)

        Returns:
            Importance ratio tensor (batch,)
        """
        if self.use_sequence_normalization:
            # Sequence-level normalization
            log_ratio = (new_log_p - old_log_p.detach()) / seq_lengths
            return torch.exp(log_ratio)
        else:
            # Standard PPO ratio (for ablation studies)
            return torch.exp(new_log_p - old_log_p.detach())

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Calculate group-based advantages for GSPO.

        GSPO uses the batch as a group and normalizes advantages:
        A_i = (R_i - mean(R_batch)) / (std(R_batch) + eps)

        Args:
            rewards: Reward tensor (batch,)
            values: Value estimates (batch,) - not used in GSPO advantage calculation

        Returns:
            Normalized advantage tensor (batch,)
        """
        # Group-based advantage (batch is the group)
        advantage = rewards - values.detach()

        # Normalize across the group
        if advantage.size(0) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return advantage

    def calculate_loss(  # noqa: ARG002
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Dummy implementation to satisfy abstract requirement.
        GSPO uses manual optimization in training_step.
        """
        return torch.tensor(0.0, device=td.device)
