"""
Dataset for precomputed baseline values.
"""

from typing import Sized, cast

import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """
    Dataset wrapping baseline values for training.
    """

    def __init__(self, dataset: Dataset, baseline: torch.Tensor):
        """
        Initialize the BaselineDataset.

        Args:
            dataset: The underlying dataset.
            baseline: Tensor of precomputed baseline values.
        """
        super().__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(cast(Sized, self.dataset)) == len(self.baseline)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieve a sample with its associated baseline value.

        Args:
            index: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and 'baseline' keys.
        """
        return {"data": self.dataset[index], "baseline": self.baseline[index]}

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(cast(Sized, self.dataset))
