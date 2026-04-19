"""
Greedy rollout baseline with significance-based updates.
"""

from __future__ import annotations

import copy
from typing import Any, Optional, cast

import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

from logic.src.constants.routing import DEFAULT_ROLLOUT_BATCH_SIZE
from logic.src.data.datasets import BaselineDataset, tensordict_collate_fn
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.data.rl_utils import safe_td_copy
from logic.src.utils.functions.rl import ensure_tensordict

from .base import Baseline

logger = get_pylogger(__name__)


class RolloutBaseline(Baseline):
    """
    Greedy rollout baseline with significance-based updates.

    Uses greedy decoding with a frozen policy copy as baseline.
    The baseline policy is updated only if the current policy outperforms it
    significantly according to a T-test.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        update_every: int = 1,
        bl_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Initialize RolloutBaseline.

        Args:
            policy: Policy to use as initial baseline (will be copied).
            update_every: Update baseline every N epochs.
            bl_alpha: Significance level for T-test to decide on updates.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.update_every = update_every
        self.bl_alpha = bl_alpha

        self.baseline_policy = None
        if policy is not None:
            self.setup(policy)

    def setup(self, policy: nn.Module):
        """Copy policy for baseline."""
        self.baseline_policy = copy.deepcopy(policy)  # type: ignore[assignment]
        if self.baseline_policy is not None:
            self.baseline_policy.eval()  # type: ignore[misc]
            for param in self.baseline_policy.parameters():
                param.requires_grad = False

    def _rollout(self, policy: nn.Module, td_or_dataset: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a batch or dataset."""
        if isinstance(td_or_dataset, Dataset):
            return self._rollout_dataset(policy, td_or_dataset, env)
        return self._rollout_batch(policy, td_or_dataset, env)

    def _rollout_dataset(self, policy: nn.Module, dataset: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a complete dataset."""
        if env is None:
            raise ValueError("Environment (env) is required for RolloutBaseline evaluation")

        # Determine strict batch size from environment
        batch_size = DEFAULT_ROLLOUT_BATCH_SIZE  # Default from constants
        if hasattr(env, "batch_size") and len(env.batch_size) > 0:
            batch_size = int(env.batch_size[0])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=0,
        )
        rewards = []
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                # Get device from policy
                device = next(policy.parameters()).device
                td_data = ensure_tensordict(batch, device)

                try:
                    if hasattr(env, "reset"):
                        td_data = env.reset(td_data)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning(f"Environment reset failed: {e}")
                    raise e

                # Padding logic for last batch
                real_size = td_data.batch_size[0]
                if real_size != batch_size:
                    pad_size = batch_size - real_size
                    # Repeat the first element to pad safely
                    padding = td_data[0].expand(pad_size)
                    padding_safe = safe_td_copy(padding)
                    td_data = torch.cat([td_data, padding_safe], 0)

                if hasattr(policy, "set_strategy"):
                    cast(Any, policy).set_strategy("greedy")
                    res = cast(Any, policy)(td_data)
                    out = {"reward": res[0]} if isinstance(res, tuple) else res
                else:
                    out = cast(Any, policy)(td_data, env, strategy="greedy")

                # Unpad rewards
                if real_size != batch_size:
                    out["reward"] = out["reward"][:real_size]

                rewards.append(out["reward"].cpu())

        if len(rewards) == 0:
            return torch.tensor([], device="cpu")
        return torch.cat(rewards)

    def _rollout_batch(self, policy: nn.Module, td: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a single batch."""
        # Note: deepcopy can be expensive, but ensure_tensordict helps standardize
        device = next(policy.parameters()).device
        td_data = ensure_tensordict(td, device)
        td_copy = copy.deepcopy(td_data)

        policy.eval()
        with torch.no_grad():
            if hasattr(policy, "set_strategy"):
                cast(Any, policy).set_strategy("greedy")
                res = cast(Any, policy)(td_copy)
                out = {"reward": res[0]} if isinstance(res, tuple) else res
            else:
                if env is None:
                    raise ValueError("Environment (env) is required for RolloutBaseline evaluation")
                out = cast(Any, policy)(td_copy, env, strategy="greedy")

        return out["reward"]

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Wrap the dataset with rollout baseline values."""
        # Compatibility: handle positional arguments if called as (policy, dataset, env)
        if isinstance(dataset, nn.Module):
            # Probably called as wrap_dataset(policy, dataset, env)
            policy_arg = dataset
            dataset_arg = policy  # Actually the 2nd arg
            env_arg = env
            dataset = dataset_arg
            policy = policy_arg
            env = env_arg
            dataset = dataset_arg

        if env is None:
            # We can't actually rollout without env, so return original
            return dataset

        # Use provided policy or fallback to baseline_policy
        p = policy if policy is not None else self.baseline_policy
        if p is None:
            return dataset

        print("Evaluating baseline on dataset...")
        bl_vals = self._rollout(p, dataset, env)
        return BaselineDataset(dataset, bl_vals.view(-1, 1))

    def eval(self, td: Any, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value.
        """
        # If we have a baseline policy, run it to get the baseline value
        if self.baseline_policy is not None:
            # Note: This is computationally expensive if done every step
            # Ideally use wrap_dataset/unwrap_batch flow
            with torch.no_grad():
                td = ensure_tensordict(td, next(iter(self.baseline_policy.parameters())).device)
                return self._rollout(self.baseline_policy, td, env)

        return torch.zeros_like(reward)

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Update baseline policy if current policy improves significantly."""
        if (epoch + 1) % self.update_every == 0:
            if (
                val_dataset is not None
                and len(val_dataset) > 0
                and self.baseline_policy is not None
                and env is not None
            ):
                # Evaluate candidate
                candidate_vals = self._rollout(policy, val_dataset, env)
                candidate_mean = candidate_vals.mean().item()

                # Evaluate baseline
                baseline_vals = self._rollout(self.baseline_policy, val_dataset, env)
                baseline_mean = baseline_vals.mean().item()

                # T-test for significance
                t_stat, p_val = stats.ttest_rel(candidate_vals.cpu().numpy(), baseline_vals.cpu().numpy())

                if candidate_mean > baseline_mean and p_val / 2 < self.bl_alpha:
                    print(f"Update baseline: {baseline_mean:.4f} -> {candidate_mean:.4f} (p={p_val / 2:.4f})")
                    self.setup(policy)
            else:
                self.setup(policy)
