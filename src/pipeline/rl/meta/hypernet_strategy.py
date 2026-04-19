"""
Hypernetwork Strategy for Meta-RL.
Wraps the HypernetworkOptimizer model to provide adaptive weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.interfaces import ITraversable
from logic.src.models.meta.hypernet.optimizer import HyperNetworkOptimizer
from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy


class HyperNetworkStrategy(WeightAdjustmentStrategy):
    """
    Weight adjustment strategy using a Hypernetwork.

    The hypernetwork observes performance metrics and time context (day)
    to generate optimal cost weights for the current situation.
    """

    def __init__(
        self,
        problem: Any,
        device: Any,
        initial_weights: Dict[str, float],
        lr: float = 1e-4,
        constraint_value: float = 1.0,
        buffer_size: int = 100,
        **kwargs,
    ):
        """Initialize Hypernetwork Meta-Learning Strategy."""
        super().__init__()
        self.cost_weight_keys = list(initial_weights.keys())
        self.device = device

        # Initialize the underlying optimizer logic
        self.optimizer = HyperNetworkOptimizer(
            cost_weight_keys=self.cost_weight_keys,
            constraint_value=constraint_value,
            device=device,
            problem=problem,
            lr=lr,
            buffer_size=buffer_size,
        )

        self.current_weights = initial_weights.copy()

        # Cache for last observed metrics to usage in propose_weights
        self.last_costs = {
            "overflows": torch.tensor(0.0),
            "kg": torch.tensor(100.0),  # Dummy default
            "km": torch.tensor(10.0),  # Dummy default
            "kg_lost": torch.tensor(0.0),
        }
        self.current_day = 0

    def propose_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Generate weights for the next step.
        """
        context = context or {}
        day = context.get("day", self.current_day)

        # HypernetworkOptimizer expects 'all_costs' dict
        # We try to use cached costs or context provided costs
        costs = context.get("costs", self.last_costs)

        # Use underlying optimizer to generate weights
        # Note: get_weights handles normalization and device movement
        self.current_weights = self.optimizer.get_weights(
            all_costs=costs, day=day, default_weights=self.current_weights
        )

        return self.current_weights

    def feedback(
        self,
        reward: float,
        metrics: Any,
        day: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Record feedback and train the hypernetwork.
        """
        if day is not None:
            self.current_day = day

        # Parse metrics
        # In MetaRLModule 'metrics' is a list: [waste_weight, cost_weight]
        # But Hypernetwork needs specific components like 'overflows', 'kg', 'km'
        # If 'metrics' passed is a dict, we can update last_costs

        if isinstance(metrics, ITraversable):
            # Update our cache of costs for next proposal
            for k in self.last_costs:
                if k in metrics:
                    val = metrics[k]
                    if isinstance(val, (int, float)):
                        val = torch.tensor(val)
                    self.last_costs[k] = val

            # For buffer update, we need a 'metrics' tensor that Hypernet expects
            # [efficiency, overflows, kg, km, kg_lost, day_progress]
            # Construct it from dictionary
            overflows = metrics.get("overflows", 0.0)
            kg = metrics.get("kg", 1.0)
            km = metrics.get("km", 1.0)
            kg_lost = metrics.get("kg_lost", 0.0)
            efficiency = kg / (km + 1e-8)
            day_progress = self.current_day / getattr(self.optimizer.hypernetwork, "n_days", 365)

            metric_tensor = torch.tensor(
                [efficiency, overflows, kg, km, kg_lost, day_progress],
                dtype=torch.float32,
                device=self.device,
            )

        elif isinstance(metrics, (list, tuple)):
            # Fallback if we only get [reward, cost]
            # We can't fully construct the detailed metrics the Hypernet expects
            # So we might just use placeholders or fail gracefully
            metric_tensor = torch.tensor(
                [reward, 0.0, 1.0, 1.0, 0.0, 0.0],  # Dummy
                dtype=torch.float32,
                device=self.device,
            )
        else:
            return

        # Prepare weights tensor
        weights_tensor = torch.tensor(
            [self.current_weights.get(k, 1.0) for k in self.cost_weight_keys],
            dtype=torch.float32,
            device=self.device,
        )

        # Update Buffer
        # performance = reward usually (or -cost)
        # HypernetworkOptimizer minimizes loss, so 'performance' should be 'lower is better' ??
        # In HypernetworkOptimizer.train:
        #   best_perf_idx = performance_batch.argmin()
        # So yes, it expects a COST (lower is better).
        # Our 'reward' is usually higher is better. So pass -reward.

        loss_val = -reward

        self.optimizer.update_buffer(
            metrics=metric_tensor,
            day=self.current_day,
            weights=weights_tensor,
            performance=loss_val,
        )

        # Train Step
        # Train periodically or every step?
        # HypernetworkOptimizer.train(epochs=10)
        # We can do one epoch per feedback or strictly periodic
        self.optimizer.train(epochs=1)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current hypernetwork weights."""
        return self.current_weights
