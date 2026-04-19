from typing import Dict, List, Optional, Tuple

import numpy as np


class StateFeatureExtractor:
    """
    Extracts features from the current state of optimization search.

    Features include:
    - Search progress (iteration percentage)
    - Solution quality (objective gap, improvement rate)
    - Solution topology (route variance, capacity utilization)
    - Stagnation metrics (iterations without improvement)
    """

    def __init__(
        self,
        progress_thresholds: Optional[Tuple[float, float]] = None,
        stagnation_thresholds: Optional[Tuple[int, int]] = None,
        diversity_thresholds: Optional[Tuple[float, float]] = None,
    ):
        self.progress_thresholds = progress_thresholds or (0.33, 0.67)
        self.stagnation_thresholds = stagnation_thresholds or (10, 30)
        self.diversity_thresholds = diversity_thresholds or (0.3, 0.7)

    def extract_features(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        iteration: int,
        max_iterations: int,
        current_cost: float,
        best_cost: float,
        stagnation_count: int,
        improvement_history: List[float],
    ) -> Dict[str, float]:
        """
        Extract normalized search state features for continuous observation.

        Args:
            routes: Current list of vehicle routes.
            dist_matrix: Problem distance matrix.
            wastes: Dictionary of node id -> waste amount.
            capacity: Vehicle capacity.
            iteration: Current search iteration.
            max_iterations: Maximum iterations allowed.
            current_cost: Total cost of the current solution.
            best_cost: Best cost found so far.
            stagnation_count: Consecutive iterations without best improvement.
            improvement_history: List of recent improvement percentages.

        Returns:
            Dictionary of named features mapping to normalized [0, 1] values.
        """
        features = {}

        # Feature 1: Search Progress (Temporal)
        features["progress"] = iteration / max(max_iterations, 1)

        # Feature 2: Solution Quality (Objective Performance)
        if best_cost > 0:
            features["optimality_gap"] = (current_cost - best_cost) / best_cost
        else:
            features["optimality_gap"] = 0.0

        # Feature 3: Improvement rate (Velocity)
        if improvement_history:
            recent = improvement_history[-10:]
            features["improvement_rate"] = float(np.mean(recent))
        else:
            features["improvement_rate"] = 0.0

        # Feature 4: Stagnation metrics (Stability)
        features["stagnation_count"] = stagnation_count
        features["stagnation_ratio"] = stagnation_count / max(iteration, 1)

        # Feature 5: Topology metrics (Geometric structure)
        if routes:
            active_routes = [r for r in routes if r]
            route_lengths = [len(r) for r in active_routes]
            route_loads = [sum(wastes.get(n, 0) for n in r) for r in active_routes]

            features["n_routes"] = len(active_routes)
            features["avg_route_length"] = float(np.mean(route_lengths)) if route_lengths else 0.0
            features["avg_utilization"] = float(np.mean([l / capacity for l in route_loads])) if route_loads else 0.0
            features["route_diversity"] = features["avg_utilization"]
        else:
            features["n_routes"] = 0
            features["avg_route_length"] = 0.0
            features["avg_utilization"] = 0.0
            features["route_diversity"] = 0.0

        return features

    def discretize_state(
        self,
        iteration: int,
        max_iterations: int,
        stagnation_count: int,
        diversity: float,
    ) -> Tuple[int, int, int]:
        """Discretize continuous state into bins for tabular RL."""
        progress = iteration / max(max_iterations, 1)

        # Phase (0, 1, 2)
        phase = 0 if progress < self.progress_thresholds[0] else (1 if progress < self.progress_thresholds[1] else 2)

        # Stagnation (0, 1, 2)
        stag = (
            0
            if stagnation_count < self.stagnation_thresholds[0]
            else (1 if stagnation_count < self.stagnation_thresholds[1] else 2)
        )

        # Diversity (0, 1, 2)
        div = 0 if diversity < self.diversity_thresholds[0] else (1 if diversity < self.diversity_thresholds[1] else 2)

        return (phase, stag, div)

    def state_to_index(self, state_tuple: Tuple[int, int, int]) -> int:
        phase, stag, div = state_tuple
        return phase * 9 + stag * 3 + div
