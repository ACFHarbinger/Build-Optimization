"""
PyTorch Hooks Utilities for WSmart-Route.

This module provides comprehensive hooks for monitoring, debugging, and optimizing
neural network training. Hooks can capture intermediate outputs, track gradients,
monitor memory usage, and analyze weight updates.

Usage Categories:
    - Attention Hooks: Capture attention weights and masks
    - Gradient Hooks: Monitor gradient flow and detect vanishing/exploding gradients
    - Activation Hooks: Track activations, dead neurons, and sparsity
    - Memory Hooks: Profile GPU memory usage and detect leaks
    - Weight Hooks: Monitor weight updates and distribution changes

Example:
    >>> from tracking.hooks import (
    ...     add_gradient_monitoring_hooks,
    ...     add_activation_capture_hooks,
    ...     add_memory_profiling_hooks,
    ... )
    >>>
    >>> # Monitor gradients during training
    >>> grad_hooks = add_gradient_monitoring_hooks(model)
    >>> loss.backward()
    >>> print_gradient_statistics(grad_hooks['gradients'])
    >>>
    >>> # Capture activations
    >>> act_hooks = add_activation_capture_hooks(model, layer_types=(nn.Linear,))
    >>> output = model(input)
    >>> for name, activation in act_hooks['activations'].items():
    ...     print(f"{name}: {activation.shape}")
    >>>
    >>> # Log hook stats to an active tracking run
    >>> register_hooks_with_run(grad_hooks, run, prefix="train/hooks")
"""

from __future__ import annotations

from typing import Any, Dict

# Activation hooks
from tracking.hooks.activation_hooks import (
    add_activation_capture_hooks,
    add_activation_sparsity_hook,
    add_activation_statistics_hook,
    add_dead_neuron_detector_hook,
    compute_activation_statistics,
    compute_sparsity_percentages,
    print_activation_summary,
)

# Attention hooks
from tracking.hooks.attention_hooks import add_attention_hooks

# Gradient hooks
from tracking.hooks.gradient_hooks import (
    add_gradient_accumulation_hook,
    add_gradient_clipping_hook,
    add_gradient_monitoring_hooks,
    add_gradient_nan_detector_hook,
    print_gradient_statistics,
)

# Memory hooks
from tracking.hooks.memory_hooks import (
    add_memory_leak_detector_hook,
    add_memory_profiling_hooks,
    estimate_model_memory,
    optimize_batch_size,
    print_memory_summary,
)

# Weight hooks
from tracking.hooks.weight_hooks import (
    add_weight_change_monitor_hook,
    add_weight_distribution_monitor,
    add_weight_norm_constraint_hook,
    add_weight_update_monitor_hook,
    analyze_weight_updates,
    compute_weight_changes,
    detect_weight_symmetry_breaking,
    print_weight_summary,
    restore_optimizer_step,
)

__all__ = [
    # Attention
    "add_attention_hooks",
    # Gradients
    "add_gradient_monitoring_hooks",
    "add_gradient_clipping_hook",
    "add_gradient_accumulation_hook",
    "add_gradient_nan_detector_hook",
    "print_gradient_statistics",
    # Activations
    "add_activation_capture_hooks",
    "add_activation_statistics_hook",
    "add_activation_sparsity_hook",
    "add_dead_neuron_detector_hook",
    "compute_activation_statistics",
    "compute_sparsity_percentages",
    "print_activation_summary",
    # Memory
    "add_memory_profiling_hooks",
    "add_memory_leak_detector_hook",
    "estimate_model_memory",
    "optimize_batch_size",
    "print_memory_summary",
    # Weights
    "add_weight_change_monitor_hook",
    "add_weight_distribution_monitor",
    "add_weight_update_monitor_hook",
    "add_weight_norm_constraint_hook",
    "compute_weight_changes",
    "detect_weight_symmetry_breaking",
    "restore_optimizer_step",
    "analyze_weight_updates",
    "print_weight_summary",
    # Tracking integration
    "register_hooks_with_run",
]


def register_hooks_with_run(hook_data: Dict[str, Any], run: Any, prefix: str = "hooks") -> None:
    """Log hook statistics to a tracking Run as params.

    Reads the result dict returned by any of the ``add_*_hook`` functions and
    serialises the numeric summaries as flat params on *run* so they appear in
    the experiment database alongside other hyper-parameters.

    Args:
        hook_data: Dict returned by one of the hook-registration helpers (must
            contain at least one of ``gradients``, ``statistics``, ``sparsity``,
            ``memory_stats``, ``dead_neurons``).
        run: An active :class:`tracking.core.run.Run` instance.
        prefix: Key prefix used when logging params (default ``"hooks"``).
    """
    if run is None:
        return

    params: Dict[str, Any] = {}

    # Gradient statistics
    gradients = hook_data.get("gradients", {})
    for layer, stats in gradients.items():
        for stat_name, val in stats.items():
            if isinstance(val, (int, float)):
                params[f"{prefix}/grad/{layer}/{stat_name}"] = round(float(val), 6)

    # Activation statistics (already-computed final stats dict)
    statistics = hook_data.get("statistics", {})
    for layer, stats in statistics.items():
        for stat_name, val in stats.items():
            if isinstance(val, (int, float)):
                params[f"{prefix}/act/{layer}/{stat_name}"] = round(float(val), 6)

    # Sparsity
    sparsity = hook_data.get("sparsity", {})
    for layer, info in sparsity.items():
        if isinstance(info, dict):
            total = info.get("total", 0)
            zeros = info.get("zeros", 0)
            pct = zeros / total if total else 0.0
        else:
            pct = float(info)
        params[f"{prefix}/sparsity/{layer}"] = round(pct, 6)

    # Memory stats - summarise peak per layer
    memory_stats = hook_data.get("memory_stats", [])
    for stat in memory_stats:
        name = stat.get("name", "unknown")
        alloc = stat.get("allocated_mb", 0)
        params[f"{prefix}/mem/{name}/allocated_mb"] = round(float(alloc), 3)

    # Dead neurons
    dead_neurons = hook_data.get("dead_neurons", {})
    for layer, count in dead_neurons.items():
        params[f"{prefix}/dead/{layer}"] = int(count)

    if params:
        run.log_params(params)
