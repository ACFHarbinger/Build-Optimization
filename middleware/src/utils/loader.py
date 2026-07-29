"""loader.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import loader
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import torch
import torch.utils.data
from tensordict import TensorDict


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    Filters out None values from samples.

    Args:
        batch: List of samples.

    Returns:
        Collated batch.
    """
    batch = [{key: val for key, val in sample.items() if val is not None} for sample in batch if sample is not None]

    # Empty lists can break collate
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)


def check_extension(filename: str, extension: str = ".pkl") -> str:
    """
    Ensures filename has the specified extension.

    Args:
        filename: Input filename.
        extension: Desired extension (e.g., '.pkl', '.td', '.pt').

    Returns:
        Filename with the specified extension.
    """
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def save_dataset(dataset: Any, filename: str) -> None:
    """
    Saves a dataset using pickle.

    Args:
        dataset: The data to save.
        filename: Target filename.

    Raises:
        Exception: If directory creation fails.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        try:
            os.makedirs(filedir, exist_ok=True)
        except Exception as e:
            raise Exception("directories to save datasets do not exist and could not be created") from e

    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


def save_td_dataset(td: TensorDict, filename: str) -> None:
    """
    Saves a TensorDict dataset.

    Args:
        td: The TensorDict to save.
        filename: Target filename.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        os.makedirs(filedir, exist_ok=True)

    torch.save(td, check_extension(filename, ".td"))


def load_td_dataset(filename: str, device: str = "cpu") -> TensorDict:
    """
    Loads a TensorDict dataset.

    Args:
        filename: The filename.
        device: Device to load onto.

    Returns:
        The loaded TensorDict.
    """
    return torch.load(check_extension(filename, ".td"), map_location=device)


def load_dataset(filename: str) -> Any:
    """
    Loads a dataset from a pickle file.

    Args:
        filename: The filename.

    Returns:
        The loaded dataset.
    """
    with open(check_extension(filename, ".pkl"), "rb") as f:
        return pickle.load(f)
