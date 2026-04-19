"""
Training/Validation step logic for RL4COLitModule.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import torch
from tensordict import TensorDict

from logic.src.constants.metrics import METRIC_MAPPING
from logic.src.interfaces import ITraversable
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.functions.rl import ensure_tensordict

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy
    from logic.src.policies.mandatory_selection import VectorizedSelector

logger = get_pylogger(__name__)


class StepMixin:
    """Mixin for training, validation, and test steps."""

    def __init__(self) -> None:
        """Initialize Class.

        Args:
            None.
        """
        # Type hints
        self.env: IEnv
        self.policy: IPolicy
        self.baseline: Any
        self.device: torch.device
        self.mandatory_selector: Optional[VectorizedSelector]
        self._current_baseline_val: Any = None
        self.last_out: Any = None

    def _apply_mandatory_selection(self, td: TensorDict) -> TensorDict:
        """
        Apply mandatory selection to determine which bins must be collected.

        Args:
            td: TensorDict with problem instance data.

        Returns:
            TensorDict with 'mandatory' mask added.
        """
        if self.mandatory_selector is None:
            return td

        # Get fill levels from the TensorDict
        fill_levels = None
        for key in ["waste", "fill_level"]:
            if key in td.keys():
                fill_levels = td[key]
                break

        if fill_levels is None:
            logger.warning("No fill levels found in TensorDict for mandatory selection")
            return td

        # Ensure fill_levels is 2D (batch_size, num_nodes)
        if fill_levels.dim() == 1:
            fill_levels = fill_levels.unsqueeze(0)

        # Get additional data for advanced selectors (lookahead, service_level)
        selector_kwargs = {}
        if "accumulation_rate" in td.keys():
            selector_kwargs["accumulation_rates"] = td["accumulation_rate"]
        if "std_deviation" in td.keys():
            selector_kwargs["std_deviations"] = td["std_deviation"]
        if "current_day" in td.keys():
            selector_kwargs["current_day"] = td["current_day"]

        # Get data needed for ManagerSelector (neural network-based selection)
        if "locs" in td.keys():
            selector_kwargs["locs"] = td["locs"]
        elif "loc" in td.keys():
            selector_kwargs["locs"] = td["loc"]

        # Waste history for temporal modeling
        if "waste_history" in td.keys():
            selector_kwargs["waste_history"] = td["waste_history"]
        elif "fill_history" in td.keys():
            selector_kwargs["waste_history"] = td["fill_history"]

        # Apply selector to get mandatory mask
        mandatory_mask = self.mandatory_selector.select(fill_levels, **selector_kwargs)

        # Ensure mandatory_mask matches the environment's node count (N+1)
        # Bins are typically customers-only in fill_levels, but mask needs depot (index 0)
        # Check num_loc in env or its generator
        num_loc = getattr(self.env, "num_loc", None)
        if num_loc is None and hasattr(self.env, "generator"):
            num_loc = getattr(self.env.generator, "num_loc", None)

        if num_loc is not None and mandatory_mask.shape[-1] == num_loc:
            # Prepend False for the depot (index 0)
            depot_mandatory = torch.zeros(*mandatory_mask.shape[:-1], 1, dtype=torch.bool, device=mandatory_mask.device)
            mandatory_mask = torch.cat([depot_mandatory, mandatory_mask], dim=-1)

        # Store in TensorDict
        td["mandatory"] = mandatory_mask

        return td

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
        td = ensure_tensordict(batch, self.device)

        # Apply mandatory selector if configured
        if self.mandatory_selector is not None:
            td = self._apply_mandatory_selection(td)

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

        # Time-based training: accumulate actions for epoch-end update
        if phase == "train" and getattr(self, "train_time", False) and "actions" in out:
            # Type ignore is safe as we checked getattr in __init__
            self._epoch_actions.append(out["actions"].detach().cpu())  # type: ignore[attr-defined]

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
