"""
Main RL4COLitModule class assembling all mixins.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from logic.src.pipeline.rl.common.baselines import WarmupBaseline, get_baseline
from logic.src.pipeline.rl.common.epoch import apply_time_step, prepare_epoch, regenerate_dataset
from logic.src.tracking.logging.pylogger import get_pylogger

from .data import DataMixin
from .optimization import OptimizationMixin
from .steps import StepMixin

if TYPE_CHECKING:
    from logic.src.configs import Config
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy
    from logic.src.models.policies.selection import VectorizedSelector

logger = get_pylogger(__name__)


class RL4COLitModule(DataMixin, OptimizationMixin, StepMixin, pl.LightningModule, ABC):
    """
    Base PyTorch Lightning module for RL training.

    This module handles:
    - Metric logging
    """

    cfg: Optional[Config] = None
    baseline: Any

    def __init__(
        self,
        env: IEnv,
        policy: IPolicy,
        baseline: Optional[str] = "rollout",
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        val_dataset_path: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        mandatory_selector: Optional[VectorizedSelector] = None,
        **kwargs,
    ):
        """
        Initialize the RL4COLitModule.

        Args:
            env: The RL environment for the problem.
            policy: The neural network policy.
            baseline: Type of baseline for variance reduction ('rollout', 'exponential', 'critic', etc.).
            optimizer: Optimizer name ('adam' or 'adamw').
            optimizer_kwargs: Keyword arguments for the optimizer.
            lr_scheduler: Learning rate scheduler name ('cosine', 'step', or None).
            lr_scheduler_kwargs: Keyword arguments for the scheduler.
            train_data_size: Number of training samples to generate per epoch.
            val_data_size: Number of validation samples.
            val_dataset_path: Optional path to a pre-saved validation dataset.
            train_dataset_path: Optional path to a pre-saved training dataset.
            batch_size: Batch size for training and validation.
            num_workers: Number of workers for data loading.
            mandatory_selector: Optional vectorized selector for mandatory bin selection.
            **kwargs: Additional keyword arguments.
        """
        pl.LightningModule.__init__(self)
        DataMixin.__init__(self)
        OptimizationMixin.__init__(self)
        StepMixin.__init__(self)

        # Explicitly save hyperparameters to handle MRO/introspection issues
        params_to_save = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "env", "policy", "mandatory_selector", "generator"]
        }
        # Avoid shadowing self.baseline (the object) with baseline (the string)
        if "baseline" in params_to_save:
            params_to_save["baseline_type"] = params_to_save.pop("baseline")
        self.save_hyperparameters(params_to_save)

        self.env = env
        self.policy = policy
        self.baseline_type = baseline
        self.train_dataset: Optional[Any] = None
        self.mandatory_selector = mandatory_selector

        # Data params
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.val_dataset_path = val_dataset_path
        self.train_dataset_path = kwargs.get("train_dataset_path")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # Time-based training parameters
        self.train_time = kwargs.get("train_time", False)
        self._epoch_actions: List[torch.Tensor] = []

        # Optimizer params
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Initialize baseline
        self._init_baseline()

    def save_weights(self, path: str):
        """
        Save model weights and hyperparameters.

        Args:
            path: Path to save the weights to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save weights and hparams to .pt file
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": self.hparams,
            },
            path,
        )

        # Save hparams to sidecar args.json
        args_path = os.path.join(os.path.dirname(path), "args.json")

        # Convert hparams to a serializable dict
        hparams_dict: Dict[str, Any] = {}
        for k, v in self.hparams.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                hparams_dict[k] = v
            else:
                hparams_dict[k] = str(v)

        try:
            with open(args_path, "w") as f:
                json.dump(hparams_dict, f, indent=4)
            logger.info(f"Saved sidecar args.json to {args_path}")
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Could not save sidecar args.json: {e}")

        # Save full Hydra config if available
        if hasattr(self, "cfg") and self.cfg is not None:
            config_path = os.path.join(os.path.dirname(path), "config.yaml")
            try:
                OmegaConf.save(config=self.cfg, f=config_path)
            except (OSError, ImportError, TypeError, ValueError) as e:
                logger.warning(f"Could not save config.yaml: {e}")

    def _init_baseline(self):
        """Initialize baseline for advantage estimation."""
        if self.baseline_type is None:
            self.baseline_type = "rollout"

        # Use baseline_type and other hparams to get the baseline object
        baseline = get_baseline(self.baseline_type, self.policy, **self.hparams)  # type: ignore[arg-type]

        # Handle warmup
        warmup_epochs = self.hparams.get("bl_warmup_epochs", 0)
        if warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, warmup_epochs)

        self.baseline = baseline

    def on_train_epoch_start(self) -> None:
        """Prepare dataset for the new epoch (e.g. wrap with baseline)."""
        assert self.train_dataset is not None
        self.train_dataset = prepare_epoch(
            self.policy,  # type: ignore[arg-type]
            self.env,
            self.baseline,
            self.train_dataset,
            self.current_epoch,
            phase="train",
        )

    def on_train_epoch_end(self):
        """Update baseline and regenerate dataset."""
        if hasattr(self.baseline, "epoch_callback"):
            # For RolloutBaseline, we pass val_dataset for the T-test
            self.baseline.epoch_callback(
                self.policy,
                self.current_epoch,
                val_dataset=self.val_dataset,
                env=self.env,
            )

        # Apply time-step updates (waste generation, collection reset)
        if getattr(self, "train_time", False) and self.train_dataset is not None:
            # Reconstruct dataset reference in case it was wrapped
            ds_ref = self.train_dataset
            if hasattr(self.baseline, "unwrap_dataset"):
                ds_ref = self.baseline.unwrap_dataset(ds_ref)

            apply_time_step(dataset=ds_ref, epoch_actions=self._epoch_actions, day=self.current_epoch, env=self.env)
            self._epoch_actions.clear()

            # Log current day properties
            td = ds_ref.data if hasattr(ds_ref, "data") else ds_ref
            if isinstance(td, dict) or hasattr(td, "get"):
                key = "waste" if "waste" in td.keys() else "fill_level"
                mean_fill = td[key].mean() if key in td.keys() else 0.0
                self.log("train/current_day", float(self.current_epoch + 1), sync_dist=True)
                self.log("train/mean_fill", mean_fill, sync_dist=True)

        # Regenerate training dataset for next epoch if configured
        if (
            self.hparams.get("regenerate_per_epoch", False)
            and self.trainer.max_epochs is not None
            and self.current_epoch < self.trainer.max_epochs - 1
            and hasattr(self.env, "generator")
        ):
            new_dataset = regenerate_dataset(self.env, self.train_data_size)
            if new_dataset is not None:
                self.train_dataset = new_dataset

            # Log generation
            if self.local_rank == 0:
                logger.info(f"Regenerated training dataset for epoch {self.current_epoch + 1}")
