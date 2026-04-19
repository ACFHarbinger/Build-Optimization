"""
SymNCO algorithm implementation.

Reference:
    Kim, M., Park, J., & Park, J. (2022).
    Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization.
    Advances in Neural Information Processing Systems, 35, 1936-1949.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.utils.tasks.losses import (
    invariance_loss,
    solution_symmetricity_loss,
)


class SymNCO(POMO):
    """
    SymNCO algorithm: REINFORCE with problem/solution symmetricity and invariance losses.

    Includes support for:
    - Data Augmentation (Dihedral/Symmetric)
    - Multi-start decoding
    - Shared baseline across starts AND augmentations
    - Consistency losses

    Reference:
        Kim, M., Park, J., Kim, J., & Park, J. (2022).
        Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization.
        NeurIPS 2022. arXiv:2205.13209
        https://arxiv.org/abs/2205.13209
    """

    def __init__(
        self,
        alpha: float = 0.2,  # weight for invariance loss
        beta: float = 1.0,  # weight for solution symmetricity loss
        **kwargs,
    ):
        """
        Initialize SymNCO module.

        Args:
            alpha: Weight for invariance loss.
            beta: Weight for solution symmetricity loss.
            **kwargs: Arguments passed to POMO.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        SymNCO shared step with symmetricity losses.
        """
        td = self.env.reset(batch)
        bs = td.batch_size[0]

        # Determine number of starts
        n_start = self.num_starts
        if n_start is None:
            n_start = self.env.get_num_starts(td) if hasattr(self.env, "get_num_starts") else td["locs"].shape[1]

        # Augmentation
        n_aug = self.num_augment
        if self.augmentation is not None:
            td = self.augmentation(td)

        # Run policy (must return init_embeds or proj_embeddings)
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
            num_starts=n_start,
        )

        # reward: [batch, n_aug, n_start]
        reward = out["reward"].view(bs, n_aug, n_start)

        # Initialize metrics for logging
        loss_ps = torch.tensor(0.0, device=reward.device)
        loss_ss = torch.tensor(0.0, device=reward.device)
        loss_inv = torch.tensor(0.0, device=reward.device)
        loss = torch.tensor(0.0, device=reward.device)
        ll = torch.tensor(0.0, device=reward.device)
        if phase == "train":
            # log_likelihood: [batch, n_aug, n_start]
            ll = out["log_likelihood"].view(bs, n_aug, n_start)

            # 1. Problem symmetricity loss (consistency across augmentations)

            # 2. Solution symmetricity loss (consistency across starts)
            # Baseline is mean across starts for each augmentation
            loss_ss = solution_symmetricity_loss(reward, ll, dim=-1)

            # 3. Invariance loss (invariant representation across augmentations)
            loss_inv_val: float | torch.Tensor
            loss_inv_val = invariance_loss(out["proj_embeddings"], n_aug) if "proj_embeddings" in out else 0.0

            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv_val

            if self.entropy_weight > 0 and "entropy" in out:
                loss = loss - self.entropy_weight * out["entropy"].mean()

            out["loss"] = loss

            # Update metrics
            best_reward, _ = reward.view(bs, -1).max(dim=-1)
            out["reward"] = best_reward
        else:
            # During val/test, we take the best across starts and augments
            max_reward_per_aug, _ = reward.max(dim=-1)
            best_reward, _ = max_reward_per_aug.max(dim=-1)
            out["reward"] = best_reward

        # Log metrics
        self.log(f"{phase}/reward", out["reward"].mean(), prog_bar=True, sync_dist=True)
        if phase == "train":
            self.log("train/loss", loss, sync_dist=True)
            self.log("train/loss_ps", loss_ps, sync_dist=True)
            self.log("train/loss_ss", loss_ss, sync_dist=True)
            self.log("train/log_likelihood", ll.mean(), sync_dist=True)
            if "proj_embeddings" in out:
                self.log("train/loss_inv", loss_inv, sync_dist=True)
            if "entropy" in out:
                self.log("train/entropy", out["entropy"].mean(), sync_dist=True)

        return out
