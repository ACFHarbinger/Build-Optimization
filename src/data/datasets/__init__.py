"""
Dataset classes for WSmart-Route.
"""

# Import from utils.data.td_utils, not from a local td_utils
from logic.utils.data.td_utils import td_kwargs, tensordict_collate_fn

from .pytorch.baseline_dataset import BaselineDataset
from .pytorch.extra_key_dataset import ExtraKeyDataset
from .pytorch.fast_gen_dataset import TensorDictDatasetFastGeneration
from .pytorch.fast_td_dataset import FastTdDataset
from .pytorch.generator_dataset import GeneratorDataset
from .pytorch.td_dataset import TensorDictDataset
from .simulation.gen_dataset import GenerativeDataset
from .simulation.np_pkl_dataset import NumpyPickleDataset
from .simulation.npz_dataset import NumpyDictDataset
from .simulation.pd_csv_dataset import PandasCsvDataset
from .simulation.pd_xlsx_dataset import PandasExcelDataset
from .simulation.sim_dataset import SimulationDataset

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
    # Simulation datasets
    "SimulationDataset",
    "NumpyDictDataset",
    "NumpyPickleDataset",
    "PandasExcelDataset",
    "PandasCsvDataset",
    "GenerativeDataset",
]
