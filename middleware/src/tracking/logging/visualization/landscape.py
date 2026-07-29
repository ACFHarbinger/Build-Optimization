"""
Loss landscape visualization tools.

Functions for computing and plotting loss landscapes.
"""

import os
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import loss_landscapes
import matplotlib.pyplot as plt
from loss_landscapes.metrics import Metric
from models.policies.local_search import vectorized_two_opt

from configs import Config
from tracking.logging.visualization.helpers import MyModelWrapper, get_batch


def imitation_loss_fn(m, x_batch, pi_target, cost_weights=None):
    """
    Computes imitation loss (log likelihood of target) for loss landscape.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        res = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False, expert_pi=pi_target)
        log_likelihood = res[1]
    return -log_likelihood.mean().item()


def rl_loss_fn(m, x_batch, cost_weights=None):
    """
    Computes RL loss (greedy cost) for loss landscape.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        model_to_call.set_strategy("greedy")
        cost, _, _, _, _ = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False)
    return cost.float().mean().item()


class ImitationMetric(Metric):
    """Metric class for imitation loss."""

    def __init__(self, x_batch, pi_target, cost_weights=None):
        super().__init__()
        self.x_batch = x_batch
        self.pi_target = pi_target
        self.cost_weights = cost_weights

    def __call__(self, model_wrapper):
        """Computes imitation loss for the current model state."""
        return imitation_loss_fn(model_wrapper, self.x_batch, self.pi_target, cost_weights=self.cost_weights)


class RLMetric(Metric):
    """Metric class for RL cost."""

    def __init__(self, x_batch, cost_weights=None):
        super().__init__()
        self.x_batch = x_batch
        self.cost_weights = cost_weights

    def __call__(self, model_wrapper):
        """Computes RL cost for the current model state."""
        return rl_loss_fn(model_wrapper, self.x_batch, cost_weights=self.cost_weights)


def plot_loss_landscape(
    model: Any,
    cfg: Config,
    output_dir: str,
    epoch: int = 0,
    size: int = 50,
    batch_size: int = 16,
    resolution: int = 10,
    span: float = 1.0,
) -> None:
    """
    Computes and plots 2D and 3D loss landscapes for both Imitation Loss and RL Cost.

    Args:
        model: The neural model.
        cfg: Root Hydra config (uses ``cfg.model.temporal_horizon`` for batch generation).
        output_dir: Directory to save plots.
        epoch: Current epoch.
        size: Graph size.
        batch_size: Batch size.
        resolution: Grid resolution.
        span: Range of perturbation.
    """
    print("Computing Loss Landscape...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    temporal_horizon: int = cfg.model.temporal_horizon

    # Generate random batch for landscape
    x_batch = get_batch(
        device,
        size=size,
        batch_size=batch_size,
        temporal_horizon=temporal_horizon,
    )

    print("Generating expert targets for landscape...")
    model.set_strategy("greedy")
    with torch.no_grad():
        _, _, _, pi, _ = model(x_batch, return_pi=True)
        x_dist = x_batch["dist"]
        if x_dist.dim() == 2:
            x_dist = x_dist.unsqueeze(0)
        if x_dist.size(0) == 1:
            x_dist = x_dist.expand(pi.size(0), -1, -1)
        pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=device), pi], dim=1)
        pi_opt = vectorized_two_opt(pi_with_depot, x_dist, max_iterations=100)
        pi_target = pi_opt[:, 1:]

    wrapped_model = MyModelWrapper(model)

    model_device = next(model.parameters()).device

    def move_dict_to_device(d, dev):
        """Recursively move dictionary values to the specified device."""
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    x_batch = move_dict_to_device(x_batch, model_device)
    pi_target = pi_target.to(model_device)

    # Imitation
    print(f"Computing Imitation Landscape on {model_device}...")

    orig_cost_weights = getattr(model, "cost_weights", None)

    # Move data to CPU for landscape computation consistency if needed by wrapper
    m_dev = torch.device("cpu")
    x_m = move_dict_to_device(x_batch, m_dev)
    pi_m = pi_target.to(m_dev)

    imitation_metric = ImitationMetric(x_m, pi_m, cost_weights=orig_cost_weights)

    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            imitation_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"Imitation Loss Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_imitation_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="viridis")  # type: ignore[attr-defined]
        plt.title(f"Imitation Loss Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_imitation_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing imitation landscape: {e}")

    # RL
    print(f"Computing RL Cost Landscape on {model_device}...")

    rl_metric = RLMetric(x_m, cost_weights=orig_cost_weights)

    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            rl_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"RL Cost Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_rl_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="magma")  # type: ignore[attr-defined]
        plt.title(f"RL Cost Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_rl_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing RL landscape: {e}")
