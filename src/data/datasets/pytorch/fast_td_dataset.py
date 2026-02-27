"""
Optimized TensorDict dataset.
"""

from typing import Any

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class FastTdDataset(Dataset):
    """
    Optimized Dataset for TensorDicts.
    """

    def __init__(self, td: TensorDict):
        """
        Initialize the FastTdDataset.

        Args:
            td: A TensorDict containing the dataset tensors.
        """
        self.data_len = td.batch_size[0]
        self.data = td

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data_len

    def __getitems__(self, index):
        """
        Retrieve samples by indices (batch getter).

        Args:
            index: Indices of the samples.

        Returns:
            TensorDict: The samples at the specified indices.
        """
        return self.data[index]

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
