"""
Manager (Neural) Selection Strategy.
"""

from typing import Optional

import torch
from torch import Tensor

from logic.src.models.meta.hrl_manager import MandatoryManager

from .base import VectorizedSelector


class ManagerSelector(VectorizedSelector):
    """
    Neural network-based mandatory selection using MandatoryManager.

    This selector wraps a trained HRL manager to make mandatory decisions
    based on learned patterns from temporal waste data and spatial context.
    The manager learns to predict which bins require collection.
    """

    def __init__(
        self,
        manager=None,
        manager_config: Optional[dict] = None,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize ManagerSelector.

        Args:
            manager: Pre-instantiated MandatoryManager. If None, creates one from config.
            manager_config: Configuration dict for creating manager if not provided.
            threshold: Probability threshold for mandatory decision.
            device: Device for computation ('cpu' or 'cuda').
        """
        self.threshold = threshold
        self.device = device

        if manager is not None:
            self.manager = manager
        else:
            config = manager_config or {}
            self.manager = MandatoryManager(
                hidden_dim=config.get("hidden_dim", 128),
                lstm_hidden=config.get("lstm_hidden", 64),
                input_dim_dynamic=config.get("history_length", 10),
                critical_threshold=config.get("critical_threshold", 0.9),
                device=device,
            )

    def select(
        self,
        fill_levels: Tensor,
        locs: Optional[Tensor] = None,
        waste_history: Optional[Tensor] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins using the neural manager.

        The manager uses spatial (locations) and temporal (waste history) features
        to predict which bins must be collected.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            locs: Node locations (batch_size, num_nodes, 2). Required.
            waste_history: Historical waste levels (batch_size, num_nodes, history).
                          If None, expands current fill levels as history.
            threshold: Optional override for probability threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes) where True = must collect.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        thresh = threshold if threshold is not None else self.threshold

        # Ensure manager is on correct device
        if next(self.manager.parameters()).device != device:
            self.manager = self.manager.to(device)

        # Prepare static features (locations)
        if locs is None:
            # If no locations provided, use dummy coordinates
            # This is a fallback but locations should be provided for proper operation
            locs = torch.zeros(batch_size, num_nodes, 2, device=device)

        # Prepare dynamic features (waste history)
        if waste_history is not None:
            dynamic = waste_history
        else:
            # Expand current fill levels as constant history
            history_len = self.manager.input_dim_dynamic
            dynamic = fill_levels.unsqueeze(-1).expand(-1, -1, history_len)

        # Prepare global features
        current_waste = fill_levels
        critical_mask = (current_waste > self.manager.critical_threshold).float()
        critical_ratio = critical_mask.mean(dim=1, keepdim=True)
        max_waste = current_waste.max(dim=1, keepdim=True)[0]
        global_features = torch.cat([critical_ratio, max_waste], dim=1)

        # Get mandatory mask from manager
        mandatory = self.manager.get_mandatory_mask(locs, dynamic, global_features, threshold=thresh)

        return mandatory

    def load_weights(self, path: str):
        """Load manager weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if "manager_state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["manager_state_dict"])
        elif "state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["state_dict"])
        else:
            self.manager.load_state_dict(checkpoint)
