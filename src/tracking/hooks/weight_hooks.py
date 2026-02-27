"""
Weight monitoring and analysis hooks.

Track weight updates, detect training issues, and visualize weight distributions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


def add_weight_change_monitor_hook(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Monitor weight changes during training to detect learning issues.

    Stores initial weights and computes change magnitude after each update.

    Args:
        model: PyTorch model.
        layer_names: Specific layers to monitor. If None, monitors all.

    Returns:
        dict: Contains 'initial_weights', 'weight_changes', and metadata.

    Example:
        >>> hook_data = add_weight_change_monitor_hook(model)
        >>> # ... train for some steps ...
        >>> changes = compute_weight_changes(model, hook_data)
        >>> for name, change in changes.items():
        ...     print(f"{name}: {change:.6f}")
    """
    initial_weights = {}

    for name, param in model.named_parameters():
        if layer_names is None or name in layer_names:
            initial_weights[name] = param.data.clone().detach()

    return {
        "initial_weights": initial_weights,
        "weight_changes": {},
    }


def compute_weight_changes(
    model: nn.Module,
    hook_data: Dict[str, Any],
    metric: str = "norm",
) -> Dict[str, float]:
    """
    Compute weight changes from initial state.

    Args:
        model: Current model state.
        hook_data: Data from add_weight_change_monitor_hook.
        metric: Change metric ('norm', 'mean_abs', 'max_abs', 'relative').

    Returns:
        dict: Weight change magnitude per layer.

    Example:
        >>> hook_data = add_weight_change_monitor_hook(model)
        >>> optimizer.step()  # Update weights
        >>> changes = compute_weight_changes(model, hook_data, metric='norm')
    """
    initial_weights = hook_data["initial_weights"]
    changes = {}

    for name, param in model.named_parameters():
        if name in initial_weights:
            initial = initial_weights[name]
            current = param.data
            diff = current - initial

            if metric == "norm":
                changes[name] = diff.norm().item()
            elif metric == "mean_abs":
                changes[name] = diff.abs().mean().item()
            elif metric == "max_abs":
                changes[name] = diff.abs().max().item()
            elif metric == "relative":
                initial_norm = initial.norm().item()
                if initial_norm > 0:
                    changes[name] = (diff.norm() / initial_norm).item()
                else:
                    changes[name] = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return changes


def add_weight_distribution_monitor(
    model: nn.Module,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """
    Monitor weight distribution statistics over time.

    Useful for detecting weight initialization issues or training instabilities.

    Args:
        model: PyTorch model.
        layer_types: Types of layers to monitor.

    Returns:
        dict: Current weight statistics.

    Example:
        >>> stats = add_weight_distribution_monitor(model)
        >>> for name, stat in stats['statistics'].items():
        ...     print(f"{name}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
    """
    statistics = {}

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and hasattr(module, "weight"):
            weight = module.weight.data
            stats = {
                "mean": weight.mean().item(),
                "std": weight.std().item(),
                "min": weight.min().item(),
                "max": weight.max().item(),
                "norm": weight.norm().item(),
                "sparsity": (weight.abs() < 1e-6).sum().item() / weight.numel(),
            }
            statistics[name] = stats

    return {"statistics": statistics}


def add_weight_update_monitor_hook(
    optimizer: torch.optim.Optimizer,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Monitor weight updates during training (momentum, learning rate effects).

    Wraps optimizer.step() to track update magnitudes.

    Args:
        optimizer: PyTorch optimizer.
        log_interval: Log statistics every N steps.

    Returns:
        dict: Contains 'update_history' and 'step_count'.

    Example:
        >>> hook_data = add_weight_update_monitor_hook(optimizer, log_interval=1)
        >>> # Training loop
        >>> for i in range(100):
        ...     loss.backward()
        ...     optimizer.step()  # Updates logged automatically
        ...     optimizer.zero_grad()
    """
    update_history = defaultdict(list)
    step_count = [0]
    original_step = optimizer.step

    # Pre-map parameters to names once outside the loop to avoid indexing object/Unknown
    # We use id(param) as a stable key
    param_names = {}

    # Simple mapping logic: associate id(tensor) with a readable string or index
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            param_names[id(param)] = f"group{i}_param{j}"

    def monitored_step(*args: Any, **kwargs: Any) -> Any:
        step_count[0] += 1
        pre_weights: Dict[int, torch.Tensor] = {}

        # Log updates only on intervals to save memory/compute
        should_log = step_count[0] % log_interval == 0

        if should_log:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        # Use id(param) instead of searching the state_dict
                        pre_weights[id(param)] = param.data.clone()

        result = original_step(*args, **kwargs)

        if should_log:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    p_id = id(param)
                    if p_id in pre_weights:
                        # Normalize update magnitude by weight magnitude for relative change
                        update_norm = (param.data - pre_weights[p_id]).norm().item()
                        update_history[param_names[p_id]].append(update_norm)

        return result

    optimizer.step = monitored_step  # type: ignore[method-assign]

    return {
        "update_history": update_history,
        "step_count": step_count,
    }


def restore_optimizer_step(optimizer: torch.optim.Optimizer, hook_data: Dict[str, Any]) -> None:
    """
    Restore original optimizer.step() function.

    Args:
        optimizer: PyTorch optimizer.
        hook_data: Data from add_weight_update_monitor_hook.
    """
    if "original_step" in hook_data:
        optimizer.step = hook_data["original_step"]  # type: ignore[method-assign]


def add_weight_norm_constraint_hook(
    model: nn.Module,
    max_norm: float = 10.0,
    layer_names: Optional[List[str]] = None,
) -> List[Any]:
    """
    Register hooks to constrain weight norms during training.

    Useful for preventing weight explosion in specific layers.

    Args:
        model: PyTorch model.
        max_norm: Maximum weight norm per layer.
        layer_names: Specific layers to constrain. If None, constrains all.

    Returns:
        list: Hook handles.

    Example:
        >>> handles = add_weight_norm_constraint_hook(model, max_norm=5.0)
        >>> # Training automatically constrains weights
    """
    handles = []

    def norm_constraint_hook(param: torch.nn.Parameter, max_norm: float) -> Callable:
        """Hook to constrain parameter norm."""

        def hook(grad: torch.Tensor) -> None:
            """
            Gradient hook that applies weight norm constraint.

            This hook is called during the backward pass. It ignores the gradient
            ('grad') and directly modifies the parameter data if its norm exceeds
            'max_norm'.

            Args:
                grad: The gradient of the parameter.
            """
            with torch.no_grad():
                norm = param.data.norm()
                if norm > max_norm:
                    param.data *= max_norm / norm

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(norm_constraint_hook(param, max_norm))
            handles.append(handle)

    return handles


def detect_weight_symmetry_breaking(
    model: nn.Module,
    threshold: float = 1e-4,
) -> Dict[str, bool]:
    """
    Detect if weights have broken symmetry from initialization.

    Symmetric weights can prevent learning. This checks if diversity exists.

    Args:
        model: PyTorch model.
        threshold: Minimum weight diversity threshold.

    Returns:
        dict: True if symmetry broken (good), False otherwise (potential issue).

    Example:
        >>> symmetry = detect_weight_symmetry_breaking(model)
        >>> for name, broken in symmetry.items():
        ...     if not broken:
        ...         print(f"⚠️  {name}: Weights may be stuck in symmetric state!")
    """
    symmetry_status = {}

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight.data

            # Check if weights have sufficient variance
            std = weight.std().item()
            symmetry_status[name] = std > threshold

    return symmetry_status


def print_weight_summary(
    weight_changes: Dict[str, float],
    weight_stats: Dict[str, Dict[str, float]],
    top_k: int = 10,
) -> None:
    """
    Print comprehensive weight analysis summary.

    Args:
        weight_changes: Weight changes from compute_weight_changes.
        weight_stats: Weight statistics from add_weight_distribution_monitor.
        top_k: Number of layers to display.

    Example:
        >>> hook_data = add_weight_change_monitor_hook(model)
        >>> # ... training ...
        >>> changes = compute_weight_changes(model, hook_data)
        >>> stats = add_weight_distribution_monitor(model)
        >>> print_weight_summary(changes, stats['statistics'])
    """
    print(f"\n{'=' * 100}")
    print(f"{'Layer Name':<40} {'Change':>10} {'Mean':>10} {'Std':>10} {'Sparsity':>10}")
    print(f"{'=' * 100}")

    # Sort by change magnitude
    sorted_layers = sorted(weight_changes.keys(), key=lambda x: weight_changes[x], reverse=True)

    for name in sorted_layers[:top_k]:
        change = weight_changes.get(name, 0.0)
        stats = weight_stats.get(name, {})

        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        sparsity = stats.get("sparsity", 0.0)

        print(f"{name:<40} {change:>10.6f} {mean:>10.4f} {std:>10.4f} {sparsity:>9.2%}")

    print(f"{'=' * 100}\n")


def analyze_weight_updates(
    update_history: Dict[str, List[float]],
    window_size: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze weight update patterns to detect training issues.

    Args:
        update_history: Update history from add_weight_update_monitor_hook.
        window_size: Window for computing moving statistics.

    Returns:
        dict: Analysis including trend, variance, and potential issues.

    Example:
        >>> hook_data = add_weight_update_monitor_hook(optimizer)
        >>> # ... training ...
        >>> analysis = analyze_weight_updates(hook_data['update_history'])
        >>> for name, metrics in analysis.items():
        ...     if metrics['is_stuck']:
        ...         print(f"⚠️  {name}: Weights may be stuck!")
    """
    analysis = {}

    for name, updates in update_history.items():
        if len(updates) < window_size:
            continue

        recent = updates[-window_size:]
        mean_update = sum(recent) / len(recent)
        variance = sum((x - mean_update) ** 2 for x in recent) / len(recent)

        # Detect potential issues
        is_stuck = mean_update < 1e-8  # Weights not changing
        is_exploding = mean_update > 1.0  # Large updates
        is_oscillating = variance > mean_update**2 * 10  # High variance

        analysis[name] = {
            "mean_update": mean_update,
            "variance": variance,
            "is_stuck": is_stuck,
            "is_exploding": is_exploding,
            "is_oscillating": is_oscillating,
        }

    return analysis
