"""
Dataset optimized for fast instantiation.
"""

from typing import Any

import torch
from logic.utils.data.td_utils import td_kwargs
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDatasetFastGeneration(Dataset):
    """
    Dataset optimized for fast instantiation (avoiding list comp).
    """

    def __init__(self, td: TensorDict):
        """
        Initialize the TensorDictDatasetFastGeneration.

        Args:
            td: A TensorDict containing the dataset tensors.
        """
        self.data = td

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitems__(self, index):
        """
        Retrieve samples by indices, returning a new TensorDict.

        Args:
            index: Indices of the samples.

        Returns:
            TensorDict: A new TensorDict with the selected samples.
        """
        return TensorDict(
            {key: item[index] for key, item in self.data.items()},
            batch_size=torch.Size([len(index)]),
            **td_kwargs,
        )

    @staticmethod
    def collate_fn(batch: Any) -> Any:
        """
        Collate function that passes batch through unchanged.

        Args:
            batch: A batch of data.

        Returns:
            The batch unchanged.
        """
        return batch
