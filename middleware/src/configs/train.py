"""
Training configuration.
"""

from dataclasses import dataclass, field


@dataclass
class GraphConfig:
    """Graph/problem instance configuration for training.

    Attributes:
        num_loc: Number of locations/items in each problem instance.
    """

    num_loc: int = 50


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        data_size: Number of instances per epoch.
        lr: Learning rate.
        precision: Training precision ('32', '16-mixed').
        seed: Random seed for reproducibility.
        train_time: Whether to use time-based training.
        graph: Graph configuration for problem instances.
    """

    n_epochs: int = 100
    batch_size: int = 256
    data_size: int = 10000
    lr: float = 1e-4
    precision: str = "32"
    seed: int = 42
    train_time: bool = False
    graph: GraphConfig = field(default_factory=GraphConfig)
