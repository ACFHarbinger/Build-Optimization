"""
Training/Validation step logic for LitModule.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import torch
from constants.metrics import METRIC_MAPPING
from tensordict import TensorDict

from interfaces import ITraversable
from tracking.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    from interfaces.env import IEnv
    from interfaces.policy import IPolicy

logger = get_pylogger(__name__)


class StepMixin:
    """Mixin for training, validation, and test steps."""

    def __init__(self):
        """Initialize Class.

        Args:
            None.
        """
        # Type hints
        self.env: IEnv
        self.policy: IPolicy
        self.baseline: Any
        self.device: torch.device
        self._current_baseline_val: Any = None
        self.last_out: Any = None

    @abstractmethod
    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[IEnv] = None,
    ) -> torch.Tensor:
        """
        Compute RL loss.

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary.
            batch_idx: Current batch index.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def shared_step(
        self,
        batch: Union[TensorDict, Dict[str, Any]],
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        Common step for train/val/test.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.
            phase: One of "train", "val", "test".

        Returns:
            Output dictionary with loss, reward, etc.
        """
        # Unwrap batch if it's from a baseline dataset
        batch, baseline_val = self.baseline.unwrap_batch(batch)

        # Move to device (crucial when pin_memory=False)
        if hasattr(batch, "to"):
            batch = cast(Any, batch).to(self.device)
        else:
            batch_obj: object = batch
            if isinstance(batch_obj, ITraversable):
                batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch_obj.items()}

        if baseline_val is not None:
            baseline_val = cast(Any, baseline_val).to(self.device)
        self._current_baseline_val = baseline_val

        # env.reset expects data on the environment's device.
        from utils.functions.rl import ensure_tensordict

        td = ensure_tensordict(batch, self.device)

        td = self.env.reset(td)

        # Run policy
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
        )

        # Get updated td from rollout (if available)
        final_td = out.get("td", td)

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx, env=self.env)

        # Log reward
        reward_mean = out["reward"].mean()
        batch_size = out["reward"].shape[0]
        # Use type: ignore because LitModule.log is known to but StepMixin is a mixin
        self.log(  # type: ignore
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log granular metrics from td if available (standardized)
        # Prioritize reward_* keys as they are typically populated at the end of rollout
        for log_key, td_keys in METRIC_MAPPING.items():
            val = None
            for k in td_keys:
                if k in final_td.keys():
                    val = final_td[k]
                    break

            if val is not None:
                # Handle negative cost/overflow convention
                if (log_key in ["cost", "overflows", "initial_overflows"]) and val.mean() < 0:
                    val = -val

                self.log(  # type: ignore
                    f"{phase}/{log_key}",
                    val.mean(),
                    sync_dist=True,
                    batch_size=batch_size,
                )

        # Store for meta-learning or logging access
        self.last_out = out

        # Log policy output diagnostics
        if "log_likelihood" in out:
            self.log(  # type: ignore
                f"{phase}/log_likelihood",
                out["log_likelihood"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        if "entropy" in out:
            self.log(  # type: ignore
                f"{phase}/entropy",
                out["entropy"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )

        return out

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            *args: Positional arguments (batch, batch_idx).
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)

        # 1. Unwrap batch if it was wrapped by baseline (e.g. RolloutBaseline)
        if hasattr(self.baseline, "unwrap_batch"):
            td, baseline_val = self.baseline.unwrap_batch(batch)
        else:
            td, baseline_val = batch, None

        # 2. Run shared step
        out = self.shared_step(td, batch_idx, phase="train")

        # 3. Calculate loss with baseline_val if available
        self._current_baseline_val = baseline_val

        return out["loss"]

    def validation_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute a single validation step.

        Args:
            *args: Positional arguments (batch, batch_idx).
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Output dictionary with metrics.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute a single test step.

        Args:
            *args: Positional arguments (batch, batch_idx).
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Output dictionary with metrics.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)
        return self.shared_step(batch, batch_idx, phase="test")
