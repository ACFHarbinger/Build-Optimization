"""
Data logic for RL4COLitModule.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, cast

from torch.utils.data import DataLoader

from logic.src.data.datasets import TensorDictDataset
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.data.td_utils import tensordict_collate_fn

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy

logger = get_pylogger(__name__)


class DataMixin:
    """Mixin for data loading logic."""

    def __init__(self) -> None:
        """Initialize Class.

        Args:
            None.
        """
        # Type hints for attributes expected from the main class
        self.env: IEnv
        self.policy: IPolicy
        self.train_data_size: int
        self.val_data_size: int
        self.val_dataset_path: Optional[str]
        self.train_dataset_path: Optional[str]
        self.batch_size: int
        self.num_workers: int
        self.persistent_workers: bool
        self.pin_memory: bool
        self.local_rank: int
        self.train_dataset: Optional[Any] = None
        self.val_dataset: Optional[Any] = None

    def setup(self, stage: str) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage: The stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit":
            # If num_workers > 0, we should generate on CPU to avoid CUDA fork issues
            # and transfer to device in shared_step.

            # Pre-generate dataset on CPU for efficiency and VRAM saving
            # This avoids the overhead of generating 1 instance at a time in __getitem__
            gen = self.env.generator
            assert gen is not None
            if hasattr(gen, "to"):
                gen = gen.to("cpu")

            if self.local_rank == 0:
                logger.info(f"Pre-generating training dataset ({self.train_data_size} instances) on CPU...")

            assert gen is not None
            if self.train_dataset_path is not None and os.path.exists(self.train_dataset_path):
                if self.local_rank == 0:
                    logger.info(f"Loading training dataset from {self.train_dataset_path}")

                self.train_dataset = TensorDictDataset.load(self.train_dataset_path)
            else:
                data = gen(batch_size=self.train_data_size)
                self.train_dataset = TensorDictDataset(data)
            assert self.train_dataset is not None
            if self.val_dataset_path is not None:
                # Load validation dataset from file (legacy parity)
                self.val_dataset = TensorDictDataset.load(self.val_dataset_path)
            else:
                if self.local_rank == 0:
                    logger.info(f"Pre-generating validation dataset ({self.val_data_size} instances) on CPU...")
                val_data = cast(Any, gen)(batch_size=self.val_data_size)
                self.val_dataset = TensorDictDataset(val_data)
                assert self.val_dataset is not None
        else:
            pass

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(
            cast(Any, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            collate_fn=tensordict_collate_fn,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.train_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.train_dataset is not None
            else False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        assert self.val_dataset is not None
        return DataLoader(
            cast(Any, self.val_dataset),
            batch_size=self.batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory if self.num_workers > 0 and self.val_dataset is not None else False,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0 and self.val_dataset is not None
            else False,
        )
