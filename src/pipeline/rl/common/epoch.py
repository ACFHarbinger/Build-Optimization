"""
Epoch-level utilities for RL training.
Includes expanded dataset handling, validation metric computation,
and time-based training (train_time)
"""

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn
from torch.utils.data import Dataset

from tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


def prepare_epoch(
    model: nn.Module,
    env: Any,
    baseline: Any,
    dataset: Dataset,
    epoch: int,
    phase: str = "train",
) -> Dataset:
    """
    Prepare dataset for a new epoch.
    Handles baseline wrapping.
    """
    # Handle baseline wrapping
    if phase == "train" and hasattr(baseline, "wrap_dataset"):
        # Unwrap dataset first to avoid nested BaselineDataset
        if hasattr(baseline, "unwrap_dataset"):
            dataset = baseline.unwrap_dataset(dataset)
        # Wrap dataset with baseline values (e.g. RolloutBaseline)
        return baseline.wrap_dataset(dataset, policy=model, env=env)
    return dataset


def regenerate_dataset(
    env: Any,
    size: int,
) -> Optional[Dataset]:
    """
    Regenerate training dataset using environment generator.
    """
    if hasattr(env, "generator"):
        # Pre-generate for efficiency
        from data.datasets import TensorDictDataset

        gen = env.generator
        if hasattr(gen, "to"):
            gen = gen.to("cpu")
        data = gen(batch_size=size)
        return TensorDictDataset(data)
    return None


def compute_validation_metrics(out: Dict, batch: TensorDict, env: Any) -> Dict[str, float]:
    """
    Compute validation metrics.
    """
    metrics: Dict[str, float] = {}
    _add_reward_metric(metrics, out)
    return metrics


def _add_reward_metric(metrics: Dict[str, float], out: Dict) -> None:
    """Add mean reward to metrics."""
    if "reward" in out:
        metrics["val/reward"] = out["reward"].mean().item()
