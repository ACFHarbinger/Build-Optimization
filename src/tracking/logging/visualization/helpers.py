"""
Visualization helper utilities.

Provides data generation and model loading functions used across visualization modules.
"""

import torch
from models.core.attention_model import AttentionModel
from models.subnets.factories.attention import AttentionComponentFactory
from torch import nn
from utils.functions import load_problem


def get_batch(device, size=50, batch_size=32, temporal_horizon=0):
    """
    Generates a random batch of VRP-like data for visualization purposes.

    Args:
        device (torch.device): Device to creating tensors on.
        size (int, optional): Graph size. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 32.
        temporal_horizon (int, optional): Temporal horizon for features. Defaults to 0.

    Returns:
        dict: Batch dictionary with keys 'depot', 'loc', 'dist', 'waste', etc.
    """
    # TODO: This should ideally use the problem's generate_instance or make_dataset
    all_coords = torch.rand(batch_size, size + 1, 2, device=device)
    depot = all_coords[:, 0, :]
    loc = all_coords[:, 1:, :]
    dist_tensor = torch.cdist(all_coords, all_coords)
    waste = torch.rand(batch_size, size, device=device)
    max_waste = torch.ones(batch_size, device=device)

    batch = {
        "depot": depot,
        "loc": loc,
        "dist": dist_tensor,
        "waste": waste,
        "max_waste": max_waste,
    }

    # Add dummy temporal features if needed
    for i in range(1, temporal_horizon + 1):
        batch[f"fill{i}"] = torch.rand(batch_size, size, device=device)

    return batch


class MyModelWrapper(nn.Module):
    """
    Wraps a model to conform to the interface expected by loss-landscapes library.
    """

    def __init__(self, model):
        """Initializes the wrapper."""
        super().__init__()
        self.model = model

    def forward(
        self,
        input,
        cost_weights=None,
        return_pi=False,
        pad=False,
        mask=None,
        expert_pi=None,
    ):
        """Forward pass of the model."""
        return self.model(input, cost_weights, return_pi, pad, mask, expert_pi)


def load_model_instance(model_path, device, size=100, problem_name="wcvrp"):
    """
    Loads a model for visualization, instantiating it with default architecture parameters.
    Note: Architecture parameters are currently hardcoded for visualization defaults.

    Args:
        model_path (str): Path to checkpoint.
        device (torch.device): Device.
        size (int, optional): Problem size. Defaults to 100.
        problem_name (str, optional): Problem name. Defaults to 'wcvrp'.

    Returns:
        nn.Module: Loaded model.
    """
    # This is a bit brittle as it assumes specific model args.
    # Ideally should load args from checkpoint or args.json
    problem = load_problem(problem_name)
    factory = AttentionComponentFactory()
    model = AttentionModel(
        embed_dim=128,
        hidden_dim=512,
        problem=problem,
        component_factory=factory,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization="instance",
        tanh_clipping=10.0,
        checkpoint_encoder=False,
        shrink_size=None,
        n_heads=8,
        n_encode_sublayers=1,
        n_decode_layers=2,
        predictor_layers=2,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    # Handle cases where checkpoint might be nested
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
