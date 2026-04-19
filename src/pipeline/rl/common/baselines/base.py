"""
Base abstract class and simple baseline implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.data.datasets import BaselineDataset
from logic.src.interfaces import ITensorDictLike
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


class Baseline(nn.Module, ABC):
    """Base class for baselines."""

    def __init__(self):
        """Initialize the baseline."""
        super().__init__()

    @abstractmethod
    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """Compute baseline value."""
        raise NotImplementedError

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch if it's wrapped with baseline values.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        if isinstance(batch, ITensorDictLike) and "data" in list(batch.keys()) and "baseline" in list(batch.keys()):
            return batch["data"], batch["baseline"]
        return batch, None

    def unwrap_dataset(self, dataset: Any) -> Any:
        """Unwrap the dataset if it's wrapped."""
        if isinstance(dataset, BaselineDataset):
            return dataset.dataset
        return dataset

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Optional callback at epoch end."""
        pass

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Optional wrap dataset."""
        return dataset

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference."""
        pass

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the optimizer."""
        return []
