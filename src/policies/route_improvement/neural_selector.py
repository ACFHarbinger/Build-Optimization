"""
Neural Operator Selector Route Improver.

Utilizes a PyTorch Neural Network (Policy) to orchestrate local search operators,
mapping the current search state to a probability distribution over available moves.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import split_tour, to_numpy, tour_distance

logger = logging.getLogger(__name__)


class OperatorSelectionPolicy(nn.Module):
    """Lightweight MLP for state-to-operator mapping."""

    def __init__(self, state_dim: int, n_operators: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_operators)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


@RouteImproverRegistry.register("neural_selector")
class NeuralSelectorRouteImprover(IRouteImprovement):
    """
    Machine-learning-augmented operator orchestrator.
    Extracts a localized state vector and predicts the optimal algorithm sequence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config or {})
        self.operators = ["steepest_two_opt", "cross_exchange", "or_opt", "ruin_recreate"]
        # In a real environment, load weights from self.config["model_weights_path"]
        self.policy_net = OperatorSelectionPolicy(state_dim=4, n_operators=len(self.operators))
        self.policy_net.eval()

    def _extract_state(
        self, current_cost: float, best_cost: float, iteration: int, max_iter: int, stagnation: int
    ) -> torch.Tensor:
        """Constructs a dimensionless state vector."""
        gap = (current_cost - best_cost) / (best_cost + 1e-6)
        progress = iteration / max_iter
        return torch.tensor([gap, progress, float(stagnation), 1.0], dtype=torch.float32)

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        dm = to_numpy(kwargs.get("distance_matrix", kwargs.get("distancesC")))
        if dm is None or not tour:
            return tour, {"algorithm": "NeuralSelectorRouteImprover"}

        max_iterations = kwargs.get("iterations", self.config.get("iterations", 50))

        current_tour = [n for n in tour]
        best_tour = [n for n in tour]
        current_cost = tour_distance(split_tour(current_tour), dm)
        best_cost = current_cost

        stagnation = 0
        trace = []

        with torch.no_grad():
            for iteration in range(max_iterations):
                # 1. State extraction and Neural Forward Pass
                state = self._extract_state(current_cost, best_cost, iteration, max_iterations, stagnation)
                probs = self.policy_net(state)

                # Sample operator based on neural probabilities
                op_idx = int(torch.multinomial(probs, 1).item())
                selected_op = self.operators[op_idx]

                # 2. Execute selected operator
                improver_cls = RouteImproverRegistry.get_route_improver_class(selected_op)
                if improver_cls is None:
                    continue
                improver = improver_cls(config=self.config)

                inner_kwargs = kwargs.copy()
                inner_kwargs["iterations"] = 1  # Single step execution

                candidate_tour, _ = improver.process(current_tour, **inner_kwargs)
                candidate_cost = tour_distance(split_tour(candidate_tour), dm)

                # 3. Evaluate
                if candidate_cost < current_cost - 1e-6:
                    current_tour = candidate_tour
                    current_cost = candidate_cost
                    stagnation = 0
                    if candidate_cost < best_cost - 1e-6:
                        best_cost = candidate_cost
                        best_tour = [n for n in candidate_tour]
                else:
                    stagnation += 1

                trace.append({"iteration": iteration, "selected": selected_op, "cost": current_cost})

        metrics: ImprovementMetrics = {
            "algorithm": "NeuralSelectorRouteImprover",
            "iterations": max_iterations,
            "trace": trace,
        }

        return best_tour, metrics
