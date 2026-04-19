"""
MDAM (Multi-Decoder Attention Model).

REINFORCE-based training wrapper for MDAM policy with custom baseline
that uses the maximum reward across paths.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.envs.base.base import RL4COEnvBase

from .policy import MDAMPolicy


def mdam_rollout(
    baseline_self: Any,
    model: nn.Module,
    env: RL4COEnvBase,
    batch_size: int = 64,
    device: str = "cpu",
    dataset: Any = None,
) -> torch.Tensor:
    """
    Custom rollout function for MDAM baseline.

    Takes the maximum reward across all paths as the baseline value.

    Args:
        baseline_self: Baseline instance (for accessing dataset).
        model: MDAM model.
        env: Environment.
        batch_size: Batch size for rollout.
        device: Device to run on.
        dataset: Dataset to use (or baseline's dataset if None).

    Returns:
        Tensor of baseline rewards (batch_size,).
    """
    dataset = baseline_self.dataset if dataset is None else dataset

    model.eval()
    model = model.to(device)

    def eval_model(batch: TensorDict) -> torch.Tensor:
        """Eval model.

        Args:
            batch (TensorDict): Description of batch.

        Returns:
            Any: Description of return value.
        """
        with torch.inference_mode():
            batch = env.reset(batch.to(device))
            result = model(batch, env, strategy="greedy")
            # Take max reward across paths
            return result["reward"].max(dim=1).values

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    rewards = torch.cat([eval_model(batch) for batch in dl], dim=0)
    return rewards


class MDAM(nn.Module):
    """
    MDAM Model with REINFORCE Training.

    Multi-Decoder Attention Model trains multiple diverse policies
    to increase the chance of finding good solutions. Uses REINFORCE
    with a custom baseline that takes the max reward across paths.

    Reference:
        Xin et al. "Multi-Decoder Attention Model with Embedding Glimpse for
        Solving Vehicle Routing Problems" (AAAI 2021)
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[MDAMPolicy] = None,
        baseline: str = "rollout",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        kl_weight: float = 0.01,
    ) -> None:
        """
        Initialize MDAM model.

        Args:
            env: Environment for training.
            policy: Pre-built MDAM policy or None to create default.
            baseline: Baseline type ('rollout', 'exponential', 'critic').
            policy_kwargs: Arguments for policy construction.
            baseline_kwargs: Arguments for baseline construction.
            kl_weight: Weight for KL divergence loss term.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        baseline_kwargs = baseline_kwargs or {}

        self.env = env
        self.kl_weight = kl_weight

        # Create policy
        if policy is None:
            self.policy = MDAMPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

        # Store baseline type for external handling
        self.baseline_type = baseline
        self.baseline_kwargs = baseline_kwargs

        # Baseline will be set up by training module if needed
        self.baseline = None

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
            **kwargs: Additional arguments.

        Returns:
            Policy output dictionary.
        """
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Calculate REINFORCE loss with KL divergence regularization.

        Args:
            td: Problem instance.
            batch: Batch data (may contain baseline info).
            policy_out: Output from policy forward pass.
            reward: Override reward (uses policy_out if None).
            log_likelihood: Override log_likelihood (uses policy_out if None).

        Returns:
            Updated policy_out with loss components.
        """
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        kl_divergence = policy_out.get("entropy", torch.tensor(0.0))

        # Baseline evaluation
        if self.baseline is not None and extra is None:
            bl_val, bl_loss = self.baseline.eval(td, reward, self.env)
        else:
            bl_val = extra if extra is not None else torch.tensor(0.0)
            bl_loss = torch.tensor(0.0)

        # Handle baseline shape for multi-path reward
        # reward: (batch, num_paths), bl_val: (batch,)
        if isinstance(bl_val, torch.Tensor) and len(bl_val.shape) > 0:
            bl_val = bl_val.unsqueeze(1)

        # Advantage = reward - baseline
        advantage = reward - bl_val

        # REINFORCE loss
        reinforce_loss = -(advantage * log_likelihood).mean()

        # Total loss with KL divergence regularization
        # Note: We *subtract* KL divergence to *encourage* diversity
        loss = reinforce_loss + bl_loss - self.kl_weight * kl_divergence

        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
                "kl_loss": kl_divergence,
            }
        )
        return policy_out

    @staticmethod
    def patch_baseline_rollout(baseline: Any) -> None:
        """
        Patch a rollout baseline to use MDAM-specific rollout.

        Call this after creating the baseline to use max-across-paths
        for baseline estimation.

        Args:
            baseline: Baseline instance to patch.
        """
        # Check if it's a warmup baseline wrapping a rollout
        if hasattr(baseline, "baseline"):
            inner = baseline.baseline
            if hasattr(inner, "rollout"):
                inner.rollout = partial(mdam_rollout, inner)
        elif hasattr(baseline, "rollout"):
            baseline.rollout = partial(mdam_rollout, baseline)
