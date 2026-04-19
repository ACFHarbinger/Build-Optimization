"""
POMO (Policy Optimization with Multiple Optima) implementation.

Reference:
    Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020).
    POMO: Policy Optimization with Multiple Optima for Neural Combinatorial Optimization.
    Advances in Neural Information Processing Systems, 33, 21188-21198.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from tensordict import TensorDict

from logic.src.data.processor.transforms import StateAugmentation
from logic.src.pipeline.rl.core.reinforce import REINFORCE


class POMO(REINFORCE):
    """
    POMO algorithm: REINFORCE with shared baseline and multi-start decoding.

    Includes support for:
    - Data Augmentation (Dihedral/Symmetric)
    - Multi-start decoding
    - Shared baseline across starts

    Reference:
        Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020).
        POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.
        NeurIPS 2020. arXiv:2010.16011
        https://arxiv.org/abs/2010.16011
    """

    def __init__(
        self,
        num_augment: int = 8,
        augment_fn: Union[str, Callable] = "dihedral8",
        first_aug_identity: bool = True,
        num_starts: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize POMO module.

        Args:
            num_augment: Number of data augmentations during inference.
            augment_fn: Augmentation function ('dihedral8', 'symmetric', or callable).
            first_aug_identity: If True, first augmentation is identity.
            num_starts: Number of starting points for multi-start decoding.
            **kwargs: Arguments passed to REINFORCE.
        """
        # POMO generally uses a shared baseline of rewards across starts
        # We set baseline to 'no' because we handle the shared baseline logic in calculate_loss
        kwargs["baseline"] = "none"
        super().__init__(**kwargs)

        self.num_augment = num_augment
        self.num_starts = num_starts
        self.augment_fn = augment_fn

        self.augmentation: Optional[StateAugmentation]
        if self.num_augment > 1:
            self.augmentation = StateAugmentation(
                num_augment=num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
            )
        else:
            self.augmentation = None

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        POMO shared step with augmentation and multi-start.
        """
        td = self.env.reset(batch)
        bs = td.batch_size[0]

        # Determine number of starts (often equals graph size)
        n_start = self.num_starts
        if n_start is None:
            # Typical for POMO: n_start = n_nodes (if possible)
            # This depends on the environment implementation
            n_start = self.env.get_num_starts(td) if hasattr(self.env, "get_num_starts") else td["locs"].shape[1]

        # Augmentation during val/test (usually not during training in basic POMO,
        # but RL4CO allows it. We'll follow RL4CO: training = no aug, unless specified)
        n_aug = self.num_augment if phase != "train" else 1
        if phase != "train" and self.augmentation is not None:
            td = self.augmentation(td)

        # Run policy with multi-start
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
            num_starts=n_start,
        )

        # Reshape rewards and log_probs if we have multiple starts/augments
        # out['reward'] is [batch * n_aug * n_start]
        # We want to unbatchify it to [batch, n_aug, n_start]

        # Note: reward is [batch_size] in normal REINFORCE
        # Here it is expanded.
        reward = out["reward"].view(bs, n_aug, n_start)

        if phase == "train":
            # Shared baseline calculation: advantage = reward - mean(reward across starts)
            # shape: [batch, 1, n_start]
            log_likelihood = out["log_likelihood"].view(bs, 1, n_start)

            # Shared baseline for each instance in batch
            baseline_val = reward.mean(dim=-1, keepdim=True)  # [batch, 1, 1]
            advantage = reward - baseline_val  # [batch, 1, n_start]

            # Standard RL4CO normalization
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Loss = -mean(advantage * log_p)
            loss = -(advantage.detach() * log_likelihood).mean()

            if self.entropy_weight > 0 and "entropy" in out:
                loss = loss - self.entropy_weight * out["entropy"].mean()

            out["loss"] = loss

            # Update metrics with the best start reward for logging
            best_reward, _ = reward.max(dim=-1)  # [batch, 1]
            out["reward"] = best_reward.squeeze(-1)  # [batch]

            # Log training diagnostics
            self.log(f"{phase}/loss", loss, sync_dist=True)
            self.log(f"{phase}/advantage", advantage.mean(), sync_dist=True)
            self.log(f"{phase}/baseline", baseline_val.mean(), sync_dist=True)
            self.log(f"{phase}/log_likelihood", log_likelihood.mean(), sync_dist=True)
            if "entropy" in out:
                self.log(f"{phase}/entropy", out["entropy"].mean(), sync_dist=True)
        else:
            # During val/test, we take the best across starts and augments
            # reward is [batch, n_aug, n_start]
            max_reward_per_aug, _ = reward.max(dim=-1)  # [batch, n_aug]
            best_reward, _ = max_reward_per_aug.max(dim=-1)  # [batch]
            out["reward"] = best_reward

        # Log metrics
        self.log(f"{phase}/reward", out["reward"].mean(), prog_bar=True, sync_dist=True)

        return out
