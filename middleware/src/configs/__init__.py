"""
Root configuration dataclasses for Build-Optimization.

Usage::

    from configs import Config
    from configs.policies import SAConfig, GAConfig
    from configs.tracking import TrackingConfig
"""

from dataclasses import dataclass, field

from .game import GameConfig, GameDataConfig
from .models.model import ModelConfig
from .optimization import OptimizationConfig
from .pipeline import DataSourceConfig, PipelineConfig
from .rl import RLConfig
from .tracking import TrackingConfig
from .train import GraphConfig, TrainConfig


@dataclass
class Config:
    """Root configuration composing all sub-configs.

    Attributes:
        game: Game profile configuration (scoring, data paths).
        pipeline: Data pipeline configuration (source, transforms).
        optimization: Optimization constraints (budget, level, time_limit).
        train: Training settings (epochs, batch_size, lr).
        rl: Reinforcement learning settings (algorithm, baseline).
        tracking: Experiment tracking and logging settings.
        model: Model configuration (type, architecture).
        seed: Global random seed.
        device: Compute device ('cpu', 'cuda', 'cuda:0').
        output_dir: Directory for outputs and checkpoints.
    """

    game: GameConfig = field(default_factory=GameConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "outputs"


__all__ = [
    "Config",
    "DataSourceConfig",
    "GameConfig",
    "GameDataConfig",
    "GraphConfig",
    "OptimizationConfig",
    "PipelineConfig",
    "RLConfig",
    "TrackingConfig",
    "TrainConfig",
]
