"""
Gradient monitoring and diagnostic hooks.

Useful for debugging vanishing/exploding gradients, monitoring gradient flow,
and identifying problematic layers during training.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


def add_gradient_monitoring_hooks(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    gradient_threshold: float = 10.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Register backward hooks to monitor gradient statistics across layers.

    Detects vanishing/exploding gradients and provides layer-wise statistics.

    Args:
        model: PyTorch model to monitor.
        layer_names: Specific layer names to monitor. If None, monitors all parameters.
        gradient_threshold: Threshold for detecting exploding gradients.
        verbose: Print warnings for abnormal gradients.

    Returns:
        dict: Contains 'gradients' (list of gradient stats) and 'handles' (hook handles).

    Example:
        >>> hook_data = add_gradient_monitoring_hooks(model)
        >>> # Train for one step
        >>> loss.backward()
        >>> # Check gradient stats
        >>> for stat in hook_data['gradients']:
        ...     print(f"{stat['name']}: mean={stat['mean']:.6f}, max={stat['max']:.6f}")
    """
    gradient_stats = []
    handles = []

    def gradient_hook(name: str) -> Callable:
        """Create a hook that captures gradient statistics for a named parameter."""

        def hook(grad: torch.Tensor) -> None:
            """
            Backward hook that captures gradient statistics.

            Args:
                grad: The gradient of the parameter.
            """
            if grad is None:
                return

            stats = {
                "name": name,
                "mean": grad.abs().mean().item(),
                "max": grad.abs().max().item(),
                "min": grad.abs().min().item(),
                "std": grad.std().item(),
                "norm": grad.norm().item(),
            }
            gradient_stats.append(stats)

            # Warn about potential issues
            if verbose:
                if stats["max"] > gradient_threshold:
                    print(f"⚠️  Exploding gradient in {name}: max={stats['max']:.2f}")
                if stats["mean"] < 1e-7:
                    print(f"⚠️  Vanishing gradient in {name}: mean={stats['mean']:.2e}")

        return hook

    # Register hooks on parameters
    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(gradient_hook(name))
            handles.append(handle)

    return {"gradients": gradient_stats, "handles": handles}


def add_gradient_clipping_hook(
    model: nn.Module,
    max_norm: float = 1.0,
    layer_names: Optional[List[str]] = None,
) -> List[Any]:
    """
    Register hooks to clip gradients per-layer during backward pass.

    Useful as an alternative to global gradient clipping when specific layers
    have gradient issues.

    Args:
        model: PyTorch model.
        max_norm: Maximum gradient norm per layer.
        layer_names: Specific layers to clip. If None, clips all.

    Returns:
        list: Hook handles.

    Example:
        >>> handles = add_gradient_clipping_hook(model, max_norm=1.0)
        >>> loss.backward()  # Gradients automatically clipped
    """
    handles = []

    def clip_hook(max_norm: float) -> Callable:
        """Create gradient clipping hook."""

        def hook(grad: torch.Tensor) -> torch.Tensor:
            """
            Backward hook that clips gradient norm.

            Args:
                grad: The gradient of the parameter.

            Returns:
                torch.Tensor: Clipped gradient.
            """
            if grad is None:
                return grad
            return torch.nn.utils.clip_grad_norm_(grad, max_norm)

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(clip_hook(max_norm))
            handles.append(handle)

    return handles


def add_gradient_accumulation_hook(
    model: nn.Module,
    accumulation_steps: int = 4,
) -> Dict[str, Any]:
    """
    Register hooks to track gradient accumulation statistics.

    Useful when using gradient accumulation to understand how gradients
    evolve across micro-batches.

    Args:
        model: PyTorch model.
        accumulation_steps: Number of accumulation steps to track.

    Returns:
        dict: Contains accumulated gradient stats and handles.

    Example:
        >>> hook_data = add_gradient_accumulation_hook(model, accumulation_steps=4)
        >>> for _ in range(4):
        ...     loss = compute_loss()
        ...     loss.backward()  # Accumulate gradients
        >>> # Check accumulated gradient stats
        >>> print(hook_data['accumulated_norms'])
    """
    accumulated_norms: Dict[str, List[float]] = {name: [] for name, _ in model.named_parameters()}
    handles = []

    def accumulation_hook(name: str) -> Callable:
        """Track gradient norm accumulation."""

        def hook(grad: torch.Tensor) -> None:
            """
            Backward hook that tracks gradient accumulation statistics.

            Args:
                grad: The gradient of the parameter.
            """
            if grad is not None:
                accumulated_norms[name].append(grad.norm().item())
                # Keep only last N accumulation steps
                if len(accumulated_norms[name]) > accumulation_steps:
                    accumulated_norms[name].pop(0)

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(accumulation_hook(name))
            handles.append(handle)

    return {
        "accumulated_norms": accumulated_norms,
        "handles": handles,
    }


def add_gradient_nan_detector_hook(
    model: nn.Module,
    raise_on_nan: bool = True,
) -> List[Any]:
    """
    Register hooks to detect NaN/Inf gradients and optionally raise errors.

    Critical for catching numerical instabilities early in training.

    Args:
        model: PyTorch model.
        raise_on_nan: If True, raises ValueError when NaN detected.

    Returns:
        list: Hook handles.

    Example:
        >>> handles = add_gradient_nan_detector_hook(model, raise_on_nan=True)
        >>> loss.backward()  # Raises error if NaN gradients occur
    """
    handles = []

    def nan_detector_hook(name: str, raise_error: bool) -> Callable:
        """Detect NaN/Inf in gradients."""

        def hook(grad: torch.Tensor) -> None:
            """
            Backward hook that detects NaN/Inf in gradients.

            Args:
                grad: The gradient of the parameter.

            Raises:
                ValueError: If NaN/Inf detected and raise_error is True.
            """
            if grad is None:
                return

            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            if has_nan or has_inf:
                msg = f"NaN/Inf detected in gradients of {name}"
                if raise_error:
                    raise ValueError(msg)
                else:
                    print(f"⚠️  {msg}")

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(nan_detector_hook(name, raise_on_nan))
            handles.append(handle)

    return handles


def remove_all_hooks(hook_data: Dict[str, Any]) -> None:
    """
    Remove all registered hooks from a hook data dictionary.

    Args:
        hook_data: Dictionary returned by hook registration functions.

    Example:
        >>> hook_data = add_gradient_monitoring_hooks(model)
        >>> # ... training ...
        >>> remove_all_hooks(hook_data)  # Clean up
    """
    if "handles" in hook_data:
        for handle in hook_data["handles"]:
            handle.remove()
        hook_data["handles"].clear()


def print_gradient_statistics(gradient_stats: List[Dict[str, Any]], top_k: int = 10) -> None:
    """
    Print formatted gradient statistics.

    Args:
        gradient_stats: List of gradient statistics from monitoring hooks.
        top_k: Number of top layers to display (by gradient magnitude).

    Example:
        >>> hook_data = add_gradient_monitoring_hooks(model)
        >>> loss.backward()
        >>> print_gradient_statistics(hook_data['gradients'], top_k=5)
    """
    if not gradient_stats:
        print("No gradient statistics available.")
        return

    # Sort by norm (descending)
    sorted_stats = sorted(gradient_stats, key=lambda x: x["norm"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"{'Layer Name':<40} {'Mean':>10} {'Max':>10} {'Norm':>10}")
    print(f"{'=' * 80}")

    for stat in sorted_stats[:top_k]:
        print(f"{stat['name']:<40} {stat['mean']:>10.6f} {stat['max']:>10.6f} {stat['norm']:>10.2f}")

    print(f"{'=' * 80}\n")
