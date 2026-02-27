"""
Memory profiling and monitoring hooks.

Track GPU memory usage, identify memory leaks, and optimize batch sizes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


def add_memory_profiling_hooks(
    model: nn.Module,
    device: Optional[torch.device] = None,
    track_allocations: bool = True,
) -> Dict[str, Any]:
    """
    Register forward hooks to track memory usage per layer.

    Args:
        model: PyTorch model to profile.
        device: Device to monitor (auto-detected if None).
        track_allocations: Track CUDA memory allocations in addition to reserved memory.

    Returns:
        dict: Contains 'memory_stats' (list of memory snapshots) and 'handles'.

    Example:
        >>> hook_data = add_memory_profiling_hooks(model)
        >>> output = model(input)
        >>> for stat in hook_data['memory_stats']:
        ...     print(f"{stat['name']}: {stat['allocated_mb']:.2f} MB")
    """
    if device is None:
        device = next(model.parameters()).device

    memory_stats = []
    handles = []

    def memory_hook(name: str, device: torch.device) -> Callable:
        """Capture memory usage after layer execution."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            """
            Forward hook that captures CUDA memory statistics.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

                stat = {
                    "name": name,
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "max_allocated_mb": max_allocated,
                }
                memory_stats.append(stat)

        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if name:  # Skip root module
            handle = module.register_forward_hook(memory_hook(name, device))
            handles.append(handle)

    return {"memory_stats": memory_stats, "handles": handles}


def add_memory_leak_detector_hook(
    model: nn.Module,
    threshold_mb: float = 100.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Detect potential memory leaks by tracking memory growth.

    Warns if memory usage increases beyond threshold between forward passes.

    Args:
        model: PyTorch model.
        threshold_mb: Memory growth threshold in MB to trigger warning.
        device: Device to monitor (auto-detected if None).

    Returns:
        dict: Contains 'memory_history' and 'handles'.

    Example:
        >>> hook_data = add_memory_leak_detector_hook(model, threshold_mb=50.0)
        >>> for _ in range(10):
        ...     output = model(input)  # Warnings printed if leaks detected
    """
    if device is None:
        device = next(model.parameters()).device

    memory_history = []
    handles = []
    initial_memory = None

    def leak_detector_hook(name: str, device: torch.device, threshold: float) -> Callable:
        """Detect memory growth."""
        nonlocal initial_memory

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            """
            Forward hook that detects memory leaks between passes.

            Args:
                module: The layer being executed.
                input: Input tensors to the layer.
                output: Output tensor from the layer.
            """
            nonlocal initial_memory

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                current_memory = torch.cuda.memory_allocated(device) / 1024**2

                if initial_memory is None:
                    initial_memory = current_memory

                growth = current_memory - initial_memory
                memory_history.append({"name": name, "memory_mb": current_memory, "growth_mb": growth})

                if growth > threshold:
                    print(f"⚠️  Potential memory leak in {name}: {growth:.2f} MB growth")

        return hook

    # Register hook on root module to track overall growth
    handle = model.register_forward_hook(leak_detector_hook("model", device, threshold_mb))
    handles.append(handle)

    return {"memory_history": memory_history, "handles": handles}


def estimate_model_memory(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate memory footprint of a model without running inference.

    Args:
        model: PyTorch model.
        input_shape: Shape of single input (without batch dimension).
        batch_size: Batch size for estimation.
        dtype: Data type of inputs.

    Returns:
        dict: Memory estimates in MB.

    Example:
        >>> memory = estimate_model_memory(model, input_shape=(100, 2), batch_size=128)
        >>> print(f"Estimated memory: {memory['total_mb']:.2f} MB")
    """
    bytes_per_element = torch.finfo(dtype).bits // 8

    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    # Estimate activation memory (rough approximation)
    # Assume activations are similar size to parameters per layer
    # num_layers = sum(1 for _ in model.modules())
    activation_memory = param_memory * batch_size * 0.5  # Rough estimate

    # Input memory
    input_memory = batch_size * torch.prod(torch.tensor(input_shape)).item() * bytes_per_element / 1024**2

    # Gradient memory (same as parameters)
    gradient_memory = param_memory

    # Optimizer state (Adam: 2x parameters)
    optimizer_memory = param_memory * 2

    return {
        "param_mb": param_memory,
        "activation_mb": activation_memory,
        "input_mb": input_memory,
        "gradient_mb": gradient_memory,
        "optimizer_mb": optimizer_memory,
        "total_mb": param_memory + activation_memory + input_memory + gradient_memory + optimizer_memory,
    }


def print_memory_summary(memory_stats: List[Dict[str, Any]], top_k: int = 10) -> None:
    """
    Print formatted memory usage summary.

    Args:
        memory_stats: Memory statistics from add_memory_profiling_hooks.
        top_k: Number of top memory-consuming layers to display.

    Example:
        >>> hook_data = add_memory_profiling_hooks(model)
        >>> output = model(input)
        >>> print_memory_summary(hook_data['memory_stats'], top_k=5)
    """
    if not memory_stats:
        print("No memory statistics available.")
        return

    # Sort by allocated memory (descending)
    sorted_stats = sorted(memory_stats, key=lambda x: x["allocated_mb"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"{'Layer Name':<40} {'Allocated':>15} {'Reserved':>15}")
    print(f"{'=' * 80}")

    for stat in sorted_stats[:top_k]:
        print(f"{stat['name']:<40} {stat['allocated_mb']:>12.2f} MB {stat['reserved_mb']:>12.2f} MB")

    total_allocated = sorted_stats[0]["allocated_mb"] if sorted_stats else 0
    total_reserved = sorted_stats[0]["reserved_mb"] if sorted_stats else 0

    print(f"{'=' * 80}")
    print(f"{'TOTAL':<40} {total_allocated:>12.2f} MB {total_reserved:>12.2f} MB")
    print(f"{'=' * 80}\n")


def optimize_batch_size(
    model: nn.Module,
    input_generator: Callable,
    initial_batch_size: int = 32,
    device: Optional[torch.device] = None,
    safety_margin: float = 0.9,
) -> int:
    """
    Find optimal batch size that maximizes GPU utilization without OOM.

    Args:
        model: PyTorch model.
        input_generator: Function that generates input of shape (batch_size, ...).
        initial_batch_size: Starting batch size.
        device: Device to test on.
        safety_margin: Use this fraction of max batch size (0.9 = 90%).

    Returns:
        int: Optimal batch size.

    Example:
        >>> def input_gen(batch_size):
        ...     return torch.randn(batch_size, 100, 2, device='cuda')
        >>> optimal_bs = optimize_batch_size(model, input_gen, initial_batch_size=32)
        >>> print(f"Optimal batch size: {optimal_bs}")
    """
    if device is None:
        device = torch.device("cuda")
    model.to(device)
    model.eval()

    # batch_size = initial_batch_size
    max_working_batch_size = initial_batch_size

    print(f"Testing batch sizes starting from {initial_batch_size}...")

    # Binary search for maximum batch size
    low, high = 1, initial_batch_size * 16
    while low <= high:
        mid = (low + high) // 2

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # Test forward pass
            with torch.no_grad():
                inputs = input_generator(mid)
                _ = model(inputs)

            torch.cuda.synchronize(device)
            max_working_batch_size = mid
            print(f"  Batch size {mid}: ✓ (Peak: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB)")

            # Try larger
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {mid}: ✗ (OOM)")
                # Try smaller
                high = mid - 1
            else:
                raise e

    optimal_batch_size = int(max_working_batch_size * safety_margin)
    print(f"\nOptimal batch size (with {safety_margin * 100:.0f}% safety margin): {optimal_batch_size}")

    return optimal_batch_size


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
