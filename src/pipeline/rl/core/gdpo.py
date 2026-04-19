"""
GDPO: Group reward-Decoupled Normalization Policy Optimization.

This module implements the GDPO algorithm which normalizes reward components
independently before aggregation to prevent reward collapse in multi-objective RL.
"""

from typing import TYPE_CHECKING, List, Optional

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv

from logic.src.pipeline.rl.core.reinforce import REINFORCE


class GDPO(REINFORCE):
    """
    Group reward-Decoupled Normalization Policy Optimization (GDPO).

    Each objective channel is normalized independently (Z-scored) across the group/batch
    before being aggregated into the final advantage signal.
    """

    weights_tensor: torch.Tensor

    def __init__(
        self,
        gdpo_objective_keys: List[str],
        gdpo_objective_weights: Optional[List[float]] = None,
        gdpo_conditional_key: Optional[str] = None,
        gdpo_renormalize: bool = True,
        **kwargs,
    ):
        """
        Initialize GDPO module.

        Args:
            gdpo_objective_keys: List of keys in the TensorDict (e.g., "reward_waste", "reward_cost")
                                 representing the raw reward components.
            gdpo_objective_weights: Weights matching the objective keys. If None, uses uniform weighting.
            gdpo_conditional_key: Optional key for conditional gating (e.g., "feasibility").
                                  If provided, rewards are only optimized if this condition is met (value=1).
            gdpo_renormalize: Whether to re-normalize the final aggregated advantage.
            **kwargs: Standard REINFORCE arguments.
        """
        super().__init__(**kwargs)
        self.objective_keys = gdpo_objective_keys
        self.objective_weights = gdpo_objective_weights
        self.conditional_key = gdpo_conditional_key
        self.renormalize_aggregated = gdpo_renormalize

        # Move weights to buffer for device handling
        if self.objective_weights is None:
            self.objective_weights = [1.0] * len(self.objective_keys)

        assert len(self.objective_weights) == len(self.objective_keys), (
            f"Weights length {len(self.objective_weights)} != Objectives length {len(self.objective_keys)}"
        )

        self.register_buffer("weights_tensor", torch.tensor(self.objective_weights, dtype=torch.float32))

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional["IEnv"] = None,
    ) -> torch.Tensor:
        """
        Compute GDPO loss using decoupled normalization.
        """
        log_likelihood = out["log_likelihood"]

        # 1. Collect Rewards
        # Shape: [batch_size, num_objectives]
        rewards_list = []
        for key in self.objective_keys:
            # We assume the environment has populated these keys in 'td' or 'out'
            # Usually rewards are in 'out' for the current step/episode end
            # but environments might write them to 'td' during step.
            # Let's check 'out' first (standard RL4CO pattern for 'reward'), then 'td'.
            if key in out:
                val = out[key]
            elif key in td.keys():
                val = td[key]
            else:
                # If a component is missing, we treat it as 0 (or error?)
                # Warning could be useful, but for now 0.
                val = torch.zeros_like(out["reward"])

            rewards_list.append(val)

        # Stack: [batch_size, num_objectives]
        raw_rewards = torch.stack(rewards_list, dim=1)

        # 2. Decoupled Normalization (Group = Batch)
        # Calculate mean and std for each objective independently across the batch
        # dim=0 is the batch dimension
        means = raw_rewards.mean(dim=0, keepdim=True)
        stds = raw_rewards.std(dim=0, keepdim=True)

        # Z-score normalization
        normalized_advantages = (raw_rewards - means) / (stds + 1e-8)

        # 3. Conditional Gating
        if self.conditional_key is not None:
            # Check if condition is met
            if self.conditional_key in out:
                condition = out[self.conditional_key]
            elif self.conditional_key in td.keys():
                condition = td[self.conditional_key]
            else:
                condition = torch.ones_like(out["reward"], dtype=torch.bool)

            # Expand condition to match [batch, num_objectives]
            condition = condition.unsqueeze(-1).expand_as(normalized_advantages)

            # If condition is False (0), the advantage for that sample effectively becomes 0
            # (or we mask it out from the gradient).
            # Standard approach: only optimize where condition is met.
            normalized_advantages = normalized_advantages * condition.float()

        # 4. Weighted Aggregation
        # Sum over objectives: [batch_size]
        aggregated_advantage = (normalized_advantages * self.weights_tensor).sum(dim=1)

        # 5. Optional Re-normalization of the final scalar
        if self.renormalize_aggregated:
            agg_mean = aggregated_advantage.mean()
            agg_std = aggregated_advantage.std()
            aggregated_advantage = (aggregated_advantage - agg_mean) / (agg_std + 1e-8)

        # 6. Policy Gradient Loss
        loss = -(aggregated_advantage.detach() * log_likelihood).mean()

        # Entropy bonus
        if self.entropy_weight > 0 and "entropy" in out:
            loss = loss - self.entropy_weight * out["entropy"].mean()

        # Logging
        self.log("train/gdpo_loss", loss, sync_dist=True)
        self.log("train/gdpo_advantage_mean", aggregated_advantage.mean(), sync_dist=True)
        self.log("train/log_likelihood", log_likelihood.mean(), sync_dist=True)
        if self.entropy_weight > 0 and "entropy" in out:
            self.log("train/entropy", out["entropy"].mean(), sync_dist=True)
        for i, key in enumerate(self.objective_keys):
            self.log(f"train/gdpo_{key}_raw_mean", raw_rewards[:, i].mean(), sync_dist=True)
            self.log(
                f"train/gdpo_{key}_norm_std",
                normalized_advantages[:, i].std(),
                sync_dist=True,
            )

        return loss
