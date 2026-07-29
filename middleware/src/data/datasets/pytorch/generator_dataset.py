"""
Dataset that generates instances on-the-fly.
"""

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class GeneratorDataset(Dataset):
    """
    Dataset that generates instances on-the-fly.
    """

    def __init__(self, generator, size: int):
        """
        Initialize the GeneratorDataset.

        Args:
            generator: A callable that generates TensorDicts when invoked.
            size: The virtual size of the dataset.
        """
        self.generator = generator
        self.size = size

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.size

    def __getitem__(self, index: int) -> TensorDict:
        """
        Generate a sample on-the-fly.

        Args:
            index: Index of the sample (not used, provided for API compatibility).

        Returns:
            TensorDict: A freshly generated sample.
        """
        return self.generator(batch_size=1)[0]
