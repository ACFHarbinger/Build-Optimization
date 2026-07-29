"""
Activation capturing and monitoring hooks.

Useful for visualizing learned representations, debugging dead neurons,
and analyzing information flow through the network.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


def add_activation_capture_hooks(
    model: nn.Module,
    layer_types: Optional[Tuple[type, ...]] = None,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Register forward hooks to capture activations from specified layers.

    Args:
        model: PyTorch model.
        layer_types: Tuple of layer types to capture (e.g., (nn.Linear, nn.Conv2d)).
                    If None, captures all layers.
        layer_names: Specific layer names to capture. If None, uses layer_types.

    Returns:
        dict: Contains 'activations' (dict mapping layer names to outputs) and 'handles'.

    Example:
        >>> hook_data = add_activation_capture_hooks(model, layer_types=(nn.Linear,))
        >>> output = model(input)
        >>> for name, activation in hook_data['activations'].items():
        ...     print(f"{name}: shape={activation.shape}, mean={activation.mean():.4f}")
    """
    activations = {}
    handles = []

    def capture_hook(name: str) -> Callable:
        """Create hook to capture layer output."""

        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """
            Forward hook that captures layer activations.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            # Store activation (detached from graph to save memory)
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                # For layers returning multiple outputs, store first one
                activations[name] = output[0].detach() if len(output) > 0 else None

        return hook

    # Register hooks
    for name, module in model.named_modules():
        should_hook = False

        # Check if this layer should be hooked
        if (
            layer_names is not None
            and name in layer_names
            or layer_types is not None
            and isinstance(module, layer_types)
            or layer_types is None
            and layer_names is None
        ):
            should_hook = True

        if should_hook and name:  # Skip empty name (root module)
            handle = module.register_forward_hook(capture_hook(name))
            handles.append(handle)

    return {"activations": activations, "handles": handles}


def add_dead_neuron_detector_hook(
    model: nn.Module,
    threshold: float = 1e-6,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """
    Detect neurons that never activate (dead ReLU problem).

    Args:
        model: PyTorch model.
        threshold: Activation threshold below which neuron is considered dead.
        layer_types: Types of layers to monitor.

    Returns:
        dict: Contains 'dead_neurons' (count per layer) and 'handles'.

    Example:
        >>> hook_data = add_dead_neuron_detector_hook(model)
        >>> for _ in range(100):  # Run multiple forward passes
        ...     output = model(input)
        >>> for name, count in hook_data['dead_neurons'].items():
        ...     print(f"{name}: {count} dead neurons")
    """
    dead_neuron_counts: Dict[str, int] = defaultdict(int)
    activation_counts: Dict[str, int] = defaultdict(int)
    handles = []

    def dead_neuron_hook(name: str, threshold: float) -> Callable:
        """Count neurons with activation below threshold."""

        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """
            Forward hook that detects dead neurons.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            if isinstance(output, torch.Tensor):
                # Count neurons with max activation below threshold
                max_activation = output.abs().max(dim=0)[0]  # Max across batch
                if max_activation.dim() > 0:  # Handle different output shapes
                    dead_count = (max_activation < threshold).sum().item()
                    dead_neuron_counts[name] += dead_count  # type: ignore[assignment]
                    activation_counts[name] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(dead_neuron_hook(name, threshold))  # type: ignore[arg-type, attr-defined]
            handles.append(handle)

    return {
        "dead_neurons": dict(dead_neuron_counts),
        "activation_counts": dict(activation_counts),
        "handles": handles,
    }


def add_activation_statistics_hook(
    model: nn.Module,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d, nn.MultiheadAttention),
) -> Dict[str, Any]:
    """
    Compute running statistics of activations (mean, std, min, max).

    Args:
        model: PyTorch model.
        layer_types: Types of layers to monitor.

    Returns:
        dict: Contains 'statistics' (dict of stats per layer) and 'handles'.

    Example:
        >>> hook_data = add_activation_statistics_hook(model)
        >>> for _ in range(10):
        ...     output = model(input)
        >>> for name, stats in hook_data['statistics'].items():
        ...     print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    """
    statistics: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"sum": 0, "sum_sq": 0, "min": float("inf"), "max": float("-inf"), "count": 0}
    )  # type: ignore[assignment]
    handles = []

    def stats_hook(name: str) -> Callable:
        """Accumulate activation statistics."""

        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """
            Forward hook that accumulates activation statistics.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            if isinstance(output, torch.Tensor):
                stats = statistics[name]
                stats["sum"] += output.sum().item()
                stats["sum_sq"] += (output**2).sum().item()
                stats["min"] = min(stats["min"], output.min().item())
                stats["max"] = max(stats["max"], output.max().item())
                stats["count"] += output.numel()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(stats_hook(name))  # type: ignore[arg-type, attr-defined]
            handles.append(handle)

    return {"statistics": dict(statistics), "handles": handles}


def compute_activation_statistics(statistics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute final statistics from accumulated values.

    Args:
        statistics: Raw statistics from add_activation_statistics_hook.

    Returns:
        dict: Computed statistics (mean, std, min, max) per layer.

    Example:
        >>> hook_data = add_activation_statistics_hook(model)
        >>> # ... forward passes ...
        >>> final_stats = compute_activation_statistics(hook_data['statistics'])
    """
    final_stats = {}
    for name, stats in statistics.items():
        count = stats["count"]
        if count > 0:
            mean = stats["sum"] / count
            variance = (stats["sum_sq"] / count) - (mean**2)
            std = variance**0.5 if variance > 0 else 0

            final_stats[name] = {
                "mean": mean,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
                "count": count,
            }
    return final_stats


def add_activation_sparsity_hook(
    model: nn.Module,
    threshold: float = 0.01,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """
    Measure sparsity of activations (percentage of near-zero activations).

    Useful for understanding network efficiency and ReLU behavior.

    Args:
        model: PyTorch model.
        threshold: Values below this are considered zero.
        layer_types: Types of layers to monitor.

    Returns:
        dict: Contains 'sparsity' (percentage per layer) and 'handles'.

    Example:
        >>> hook_data = add_activation_sparsity_hook(model, threshold=0.01)
        >>> output = model(input)
        >>> for name, sparsity in hook_data['sparsity'].items():
        ...     print(f"{name}: {sparsity:.2%} sparse")
    """
    sparsity_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"zeros": 0, "total": 0})  # type: ignore[assignment]
    handles = []

    def sparsity_hook(name: str, threshold: float) -> Callable:
        """Count near-zero activations."""

        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """
            Forward hook that measures activation sparsity.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            if isinstance(output, torch.Tensor):
                stats = sparsity_stats[name]
                stats["zeros"] += (output.abs() < threshold).sum().item()  # type: ignore[assignment]
                stats["total"] += output.numel()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(sparsity_hook(name, threshold))  # type: ignore[arg-type, attr-defined]
            handles.append(handle)

    return {"sparsity": dict(sparsity_stats), "handles": handles}


def compute_sparsity_percentages(sparsity_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Compute sparsity percentages from raw counts.

    Args:
        sparsity_stats: Raw counts from add_activation_sparsity_hook.

    Returns:
        dict: Sparsity percentage per layer.
    """
    return {
        name: (stats["zeros"] / stats["total"]) if stats["total"] > 0 else 0.0 for name, stats in sparsity_stats.items()
    }


def print_activation_summary(
    statistics: Dict[str, Dict[str, float]],
    sparsity: Optional[Dict[str, float]] = None,
    dead_neurons: Optional[Dict[str, int]] = None,
) -> None:
    """
    Print comprehensive activation analysis summary.

    Args:
        statistics: Activation statistics from compute_activation_statistics.
        sparsity: Sparsity percentages from compute_sparsity_percentages.
        dead_neurons: Dead neuron counts from add_dead_neuron_detector_hook.

    Example:
        >>> stats_hook = add_activation_statistics_hook(model)
        >>> sparsity_hook = add_activation_sparsity_hook(model)
        >>> # ... forward passes ...
        >>> stats = compute_activation_statistics(stats_hook['statistics'])
        >>> sparsity = compute_sparsity_percentages(sparsity_hook['sparsity'])
        >>> print_activation_summary(stats, sparsity)
    """
    print(f"\n{'=' * 100}")
    print(f"{'Layer Name':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Sparsity':>10}")
    print(f"{'=' * 100}")

    for name in sorted(statistics.keys()):
        stats = statistics[name]
        sparsity_val = sparsity.get(name, 0.0) if sparsity else 0.0

        print(
            f"{name:<40} "
            f"{stats['mean']:>10.4f} "
            f"{stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} "
            f"{stats['max']:>10.4f} "
            f"{sparsity_val:>9.2%}"
        )

    if dead_neurons:
        print(f"\n{'=' * 100}")
        print("Dead Neurons (per layer):")
        print(f"{'=' * 100}")
        for name, count in sorted(dead_neurons.items()):
            print(f"  {name:<40}: {count} dead neurons")

    print(f"{'=' * 100}\n")


def remove_all_hooks(hook_data: Dict[str, Any]) -> None:
    """
    Remove all registered hooks.

    Args:
        hook_data: Dictionary returned by hook registration functions.
    """
    if "handles" in hook_data:
        for handle in hook_data["handles"]:
            handle.remove()
        hook_data["handles"].clear()
