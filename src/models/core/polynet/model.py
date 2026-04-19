"""
PolyNet Model.

REINFORCE-based training with Poppy loss for diverse solution strategies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.envs.base.base import RL4COEnvBase

from .policy import PolyNetPolicy


class PolyNet(nn.Module):
    """
    PolyNet Model with REINFORCE + Poppy Loss.

    Learns K diverse solution strategies using binary vector conditioning
    and Poppy loss for training. Uses shared baseline only.

    Reference:
        Hottung et al. "PolyNet: Learning Diverse Solution Strategies for
        Neural Combinatorial Optimization" (2024)

        Grinsztajn et al. "Population-Based Reinforcement Learning for
        Combinatorial Optimization" (ICLR 2022) - Poppy loss
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[PolyNetPolicy] = None,
        k: int = 128,
        val_num_solutions: int = 800,
        encoder_type: str = "AM",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        num_augment: int = 8,
        **kwargs,
    ) -> None:
        """
        Initialize PolyNet model.

        Args:
            env: Environment for training.
            policy: Pre-built policy or None to create default.
            k: Number of strategies to learn.
            val_num_solutions: Number of solutions during validation.
            encoder_type: Encoder type ("AM" or "MatNet").
            policy_kwargs: Arguments for policy construction.
            num_augment: Number of data augmentations.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}

        self.env = env
        self.k = k
        self.val_num_solutions = val_num_solutions
        self.num_augment = num_augment

        # Create policy if not provided
        if policy is None:
            self.policy = PolyNetPolicy(
                k=k,
                encoder_type=encoder_type,
                env_name=env.name,
                **policy_kwargs,
            )
        else:
            self.policy = policy

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through policy.

        Args:
            td: Problem instance TensorDict.
            env: Environment (uses self.env if None).
            phase: Training phase.

        Returns:
            Policy output dictionary.
        """
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Shared step for training/validation/testing.

        Args:
            batch: Batch of problem instances.
            batch_idx: Batch index.
            phase: Current phase.

        Returns:
            Dictionary with loss and metrics.
        """
        td = self.env.reset(batch)

        # Get number of solutions based on phase
        num_solutions = self.k if phase == "train" else self.val_num_solutions

        # Apply augmentation if training
        if phase == "train" and self.num_augment > 1:
            # Repeat for augmentation
            td = cast(TensorDict, self.env.reset(batch)).repeat_interleave(self.num_augment, dim=0)  #

        # Forward pass
        out = self.policy(
            td=td,
            env=self.env,
            phase=phase,
            return_actions=True,
            num_starts=num_solutions,
        )

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        return out

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Poppy loss for diverse solution learning.

        Poppy loss uses the mean reward across solutions as baseline,
        encouraging exploration of diverse strategies.

        Args:
            td: Problem instance.
            batch: Batch data.
            policy_out: Output from policy forward pass.
            reward: Override reward (uses policy_out if None).
            log_likelihood: Override log_likelihood (uses policy_out if None).

        Returns:
            Updated policy_out with loss components.
        """
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Reshape for multi-solution: (batch * k) -> (batch, k)
        if reward.dim() == 1:
            from logic.src.utils.decoding import unbatchify

            reward = unbatchify(reward, self.k)
            log_likelihood = unbatchify(log_likelihood, self.k)

        # Poppy loss: use mean reward as shared baseline
        # advantage = r_i - mean(r_j for all j)
        baseline = reward.mean(dim=-1, keepdim=True)
        advantage = reward - baseline

        # REINFORCE loss
        loss = -(advantage * log_likelihood).mean()

        # Track best reward
        max_reward = reward.max(dim=-1).values.mean()

        policy_out.update(
            {
                "loss": loss,
                "max_reward": max_reward,
                "baseline": baseline.mean(),
            }
        )

        return policy_out

    def rollout(
        self,
        dataset: Any,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Rollout policy on dataset.

        Args:
            dataset: Dataset to evaluate.
            batch_size: Batch size for evaluation.
            device: Device to run on.

        Returns:
            Tensor of rewards.
        """
        self.eval()
        self.to(device)

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

        rewards = []
        for batch in dl:
            with torch.inference_mode():
                batch = self.env.reset(batch.to(device))
                result = self.policy(batch, self.env, phase="val")
                rewards.append(result["reward"].max(dim=-1).values)

        return torch.cat(rewards, dim=0)
