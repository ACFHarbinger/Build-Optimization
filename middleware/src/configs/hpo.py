"""Hyperparameter-optimization configuration for the inherited RL pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration.

    Attributes:
        method: HPO method name (e.g. 'dehbo', 'optuna', 'hyp_rl').
        metric: Optimization metric ('reward', 'cost').
        n_trials: Number of HPO trials.
        n_epochs_per_trial: Training epochs per trial.
        num_workers: Number of parallel workers.
        search_space: Typed search-space spec consumed by ``pipeline.rl.hpo``.
    """

    method: str = "optuna"
    metric: str = "reward"
    n_trials: int = 0
    n_epochs_per_trial: int = 10
    num_workers: int = 4
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
