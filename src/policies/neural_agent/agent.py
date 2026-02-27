"""
Main Neural Agent class assembling mixins.
"""

from __future__ import annotations

from logic.src.tracking.viz_mixin import PolicyVizMixin

from .batch import BatchMixin
from .simulation import SimulationMixin


class NeuralAgent(PolicyVizMixin, BatchMixin, SimulationMixin):
    """
    Agent interface between simulator/environment and neural routing models.

    Handles model inference, hierarchical decision-making (HRL), and
    post-processing for waste collection routing.

    Attributes:
        model: The neural routing model (AttentionModel, etc.)
        problem: Problem instance (VRPP, WCVRP, etc.) for cost calculation
    """

    def __init__(self, model):
        """
        Initializes the NeuralAgent.

        Args:
            model: The neural routing model (e.g., AttentionModel).
        """
        self.model = model
        self.problem = model.problem
