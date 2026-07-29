"""
Standard TensorDict dataset.
"""

import torch
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    """
    Dataset compatible with TensorDicts.
    """

    def __init__(self, td: TensorDict):
        """
        Initialize the TensorDictDataset.

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
        return self.data.batch_size[0]

    def __getitem__(self, index):
        """
        Retrieve a sample by index.

        Args:
            index: Index of the sample.

        Returns:
            TensorDict: The sample at the specified index.
        """
        return self.data[index]

    @staticmethod
    def load(path: str):
        """
        Load a TensorDictDataset from disk.

        Args:
            path: File path to load from.

        Returns:
            TensorDictDataset: The loaded dataset.
        """
        return TensorDictDataset(torch.load(path))

    def save(self, path: str):
        """
        Save the dataset to disk.

        Args:
            path: File path to save to.
        """
        torch.save(self.data, path)
