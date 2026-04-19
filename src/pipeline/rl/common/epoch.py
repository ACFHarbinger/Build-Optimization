"""
Epoch-level utilities for RL training.
Includes expanded dataset handling, validation metric computation,
and time-based training (train_time) logic.
"""

from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import Dataset

from logic.src.data.datasets import TensorDictDataset
from logic.src.tracking.logging.pylogger import get_pylogger

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
    Handles baseline wrapping and temporal metadata injection.
    """
    # 1. Inject temporal metadata (current_day) for time-based training
    if hasattr(dataset, "data") and isinstance(dataset.data, TensorDict):
        td: TensorDict = dataset.data
        # If we are in training or if the dataset already has a current_day key (continuity)
        if "current_day" in td.keys() or getattr(model, "train_time", False):
            td.set("current_day", torch.full(td.batch_size, epoch, device=td.device))

    # 2. Handle baseline wrapping
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
        gen = env.generator
        if hasattr(gen, "to"):
            gen = gen.to("cpu")
        data = gen(batch_size=size)
        return TensorDictDataset(data)
    return None


def compute_validation_metrics(out: Dict, batch: TensorDict, env: Any) -> Dict[str, float]:
    """
    Compute rich validation metrics beyond simple reward.
    """
    metrics: Dict[str, float] = {}
    _add_reward_metric(metrics, out)
    _add_costs_metrics(metrics, out, batch, env)
    _add_efficiency_metric(metrics)
    _add_overflow_metrics(metrics, out, batch, env)
    return metrics


def _add_reward_metric(metrics: Dict[str, float], out: Dict) -> None:
    """Add mean reward to metrics."""
    if "reward" in out:
        metrics["val/reward"] = out["reward"].mean().item()


def _add_costs_metrics(metrics: Dict[str, float], out: Dict, batch: TensorDict, env: Any) -> None:
    """Add detailed cost breakdown from environment if available."""
    if hasattr(env, "get_costs") and "actions" in out:
        costs = env.get_costs(batch, out["actions"])
        for key, val in costs.items():
            if isinstance(val, torch.Tensor):
                metrics[f"val/{key}"] = val.float().mean().item()


def _add_efficiency_metric(metrics: Dict[str, float]) -> None:
    """Compute efficiency ratio (kg/km) from waste and distance."""
    if "val/efficiency" not in metrics and "val/waste" in metrics and "val/dist" in metrics:
        avg_waste = metrics["val/waste"]
        avg_dist = metrics["val/dist"]
        if avg_dist > 1e-6:
            metrics["val/efficiency"] = avg_waste / avg_dist


def _add_overflow_metrics(metrics: Dict[str, float], out: Dict, batch: TensorDict, env: Any) -> None:
    """Add constraint violation counts (overflows) if available."""
    if hasattr(env, "get_num_overflows") and "actions" in out:
        overflows = env.get_num_overflows(batch, out["actions"])
        metrics["val/overflows"] = overflows.float().mean().item()


def _build_visited_mask(
    epoch_actions: List[torch.Tensor], batch_size: int, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """Build a boolean mask of visited nodes from accumulated epoch actions."""
    visited_mask = torch.zeros((batch_size, num_nodes + 1), dtype=torch.bool, device=device)
    if not epoch_actions:
        return visited_mask

    try:
        all_actions = torch.cat(epoch_actions, dim=0).to(device)
        # Handle padding if actions list was larger than dataset size
        # (e.g. from rollout baseline wrap_dataset)
        if all_actions.shape[0] > batch_size:
            all_actions = all_actions[:batch_size]

        visited_mask.scatter_(1, all_actions.long(), True)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Could not build visited mask: {e}")

    # Depot (0) doesn't have waste, ignore it
    visited_mask[:, 0] = False
    return visited_mask


def _get_next_day_waste(
    td: TensorDict,
    current_fill: torch.Tensor,
    day: int,
    env: Any,
    batch_size: int,
    device: torch.device,
    key: str = "waste",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Determine the next day's waste from pre-generated data or on-the-fly generation."""
    next_day_waste = torch.zeros_like(current_fill)
    if generator is None:
        generator = torch.Generator(device=device)

    full_key = f"_{key}_full"
    if full_key in td.keys() and td[full_key].dim() == 3:
        # Use the sequence stored in the hidden key
        total_days = td[full_key].shape[1]
        next_day_idx = min(day + 1, total_days - 1)
        next_day_waste = td[full_key][:, next_day_idx, :]
    elif key in td.keys() and td[key].dim() == 3:
        # Pre-generated 3D dataset: [bs, num_days, num_loc]
        total_days = td[key].shape[1]
        next_day_idx = min(day + 1, total_days - 1)
        next_day_waste = td[key][:, next_day_idx, :]
    elif hasattr(env, "generator"):
        # On-the-fly dataset (2D waste)
        gen = env.generator
        try:
            # Generate raw fill levels for the next step (following builder logic)
            if hasattr(gen, "to"):
                gen = gen.to("cpu")
            fresh_fill = gen._generate_fill_levels([batch_size]).to(device)

            # Apply stochastic noise if it's SCWCVRP
            has_noise = env.name == "scwcvrp"
            noise_variance = getattr(gen, "noise_variance", 0.0)
            noise_mean = getattr(gen, "noise_mean", 0.0)
            if has_noise:
                noise = torch.normal(
                    mean=float(noise_mean),
                    std=float(noise_variance) ** 0.5,
                    size=fresh_fill.size(),
                    device=device,
                    generator=generator,
                )
                noisy_waste = (fresh_fill + noise).clamp(min=0.0, max=float(getattr(gen, "capacity", 1.0)))
                next_day_waste = noisy_waste
            else:
                next_day_waste = fresh_fill
        except Exception as e:
            logger.warning(f"Failed to generate fresh waste on the fly: {e}")

    return next_day_waste


def apply_time_step(dataset: Dataset, epoch_actions: List[torch.Tensor], day: int, env: Any) -> Dataset:
    """
    Applies the time-step progression to the training dataset for an epoch.
    - Marks visited nodes (collected bins) to have 0 waste.
    - Adds the next day's waste to the uncollected bins' carryover.

    Args:
        dataset: The Dataset (typically TensorDictDataset) to update.
        epoch_actions: List of action tensors taken during the epoch.
        day: Current day index (zero-based).
        env: The environment, used for generation rate or fresh generation.

    Returns:
        The mutated dataset.
    """
    if not hasattr(dataset, "data"):
        return dataset

    td: TensorDict = dataset.data
    if td is None or not isinstance(td, TensorDict):
        return dataset

    batch_size = td.batch_size[0]
    num_nodes = td["locs"].shape[1] - 1  # Exclude depot
    device = td.device

    key = "waste" if "waste" in list(td.keys()) else "fill_level"
    if key not in td.keys():
        return dataset

    current_fill = td[key]

    # If the fill is 3D, it's likely a fresh dataset load.
    # We move it to a hidden key and use the current day's slice for the 2D state.
    if current_fill.dim() == 3:
        full_key = f"_{key}_full"
        if full_key not in td.keys():
            td.set(full_key, current_fill)
        current_fill = current_fill[:, day, :]

    # 1. Reset visited bins to 0
    visited_mask = _build_visited_mask(epoch_actions, batch_size, num_nodes, device)

    if current_fill.shape == visited_mask.shape:
        current_fill[visited_mask] = 0.0
    elif current_fill.shape[1] == num_nodes:
        # If fill doesn't include depot
        current_fill[visited_mask[:, 1:]] = 0.0

    # 2. Get next day waste
    next_day_waste = _get_next_day_waste(td, current_fill, day, env, batch_size, device, key=key)

    # 3. Update fill
    new_fill = current_fill + next_day_waste

    # 4. Clamp non-negative
    new_fill = torch.clamp(new_fill, min=0.0)

    td[key] = new_fill
    if "waste" in td.keys() and key != "waste" and td["waste"].dim() == 2:
        td["waste"] = new_fill

    # Always ensure current_day is set
    td.set("current_day", torch.full(td.batch_size, day + 1, device=device))

    logger.info(f"Time Training: Updated dataset for Day {day + 1}. Mean fill: {new_fill.mean():.3f}")

    return dataset
