"""
Dataset with extra key/value pairs.
"""

from typing import Sized, cast

import torch
from torch.utils.data import Dataset


class ExtraKeyDataset(Dataset):
    """
    Dataset that includes an extra key/value pair (e.g., for baseline rewards).
    """

    def __init__(self, dataset: Dataset, extra: dict[str, torch.Tensor]):
        """
        Initialize the ExtraKeyDataset.

        Args:
            dataset: The underlying dataset.
            extra: Dictionary of extra tensors to include (e.g., {'baseline': ...}).
        """
        super().__init__()
        self.dataset = dataset
        self.extra = extra
        # Validate lengths
        for k, v in extra.items():
            assert len(cast(Sized, dataset)) == len(v), (
                f"Length mismatch for key {k}: {len(cast(Sized, dataset))} vs {len(v)}"
            )

    def __getitem__(self, index: int) -> dict:
        """
        Retrieve a sample with extra keys.

        Args:
            index: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and extra keys.
        """
        item = {"data": self.dataset[index]}
        for k, v in self.extra.items():
            item[k] = v[index]
        return item

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns:
            int: Number of samples.
        """
        return len(cast(Sized, self.dataset))
