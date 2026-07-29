"""
Dataset classes for WSmart-Route.

Heavy PyTorch / tensordict imports are guarded so that the games subpackage
(``data.datasets.games``) can be used without a working torch installation.
"""

try:
    from utils.td_utils import td_kwargs, tensordict_collate_fn

    from .pytorch.baseline_dataset import BaselineDataset
    from .pytorch.extra_key_dataset import ExtraKeyDataset
    from .pytorch.fast_gen_dataset import TensorDictDatasetFastGeneration
    from .pytorch.fast_td_dataset import FastTdDataset
    from .pytorch.generator_dataset import GeneratorDataset
    from .pytorch.td_dataset import TensorDictDataset
except Exception:
    pass

__all__ = [
    "td_kwargs",
    "tensordict_collate_fn",
    # PyTorch datasets
    "BaselineDataset",
    "ExtraKeyDataset",
    "TensorDictDatasetFastGeneration",
    "FastTdDataset",
    "GeneratorDataset",
    "TensorDictDataset",
]
