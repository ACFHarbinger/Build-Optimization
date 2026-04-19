"""
Ruin-and-Recreate operators with adaptive selection mechanism.

This module implements the destroy/repair paradigm integrated with HGS,
including adaptive operator weight management using scoring feedback.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators import destroy_ruin as destroy_operators
from logic.src.policies.helpers.operators import recreate_repair as repair_operators
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search import Individual
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution import evaluate

from .params import HGSRRParams


class AdaptiveOperatorManager:
    """
    Manages adaptive selection of destroy/repair operators using a scoring mechanism.

    Attributes:
        destroy_weights (Dict[str, float]): Current weights for destroy operators.
        repair_weights (Dict[str, float]): Current weights for repair operators.
        destroy_scores (Dict[str, float]): Accumulated scores for destroy operators.
        repair_scores (Dict[str, float]): Accumulated scores for repair operators.
        destroy_counts (Dict[str, int]): Usage counts for destroy operators.
        repair_counts (Dict[str, int]): Usage counts for repair operators.
    """

    def __init__(
        self,
        destroy_operators: List[str],
        repair_operators: List[str],
        reaction_factor: float = 0.1,
        decay_parameter: float = 0.95,
    ):
        """
        Initialize the adaptive operator manager.

        Args:
            destroy_operators: List of destroy operator names.
            repair_operators: List of repair operator names.
            reaction_factor: Rate of weight updates (0.0-1.0).
            decay_parameter: Exponential decay rate for weights (0.0-1.0).
        """
        self.reaction_factor = reaction_factor
        self.decay_parameter = decay_parameter

        # Initialize uniform weights
        self.destroy_weights = {op: 1.0 for op in destroy_operators}
        self.repair_weights = {op: 1.0 for op in repair_operators}

        # Score tracking
        self.destroy_scores = {op: 0.0 for op in destroy_operators}
        self.repair_scores = {op: 0.0 for op in repair_operators}

        # Usage counts
        self.destroy_counts = {op: 0 for op in destroy_operators}
        self.repair_counts = {op: 0 for op in repair_operators}

        # Segment control
        self.segment_size = 100
        self.current_iteration = 0

    def select_operators(self, rng: random.Random) -> Tuple[str, str]:
        """
        Select destroy and repair operators using roulette wheel selection.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (destroy_operator_name, repair_operator_name).
        """
        destroy_op = self._roulette_wheel(self.destroy_weights, rng)
        repair_op = self._roulette_wheel(self.repair_weights, rng)

        self.destroy_counts[destroy_op] += 1
        self.repair_counts[repair_op] += 1

        return destroy_op, repair_op

    def update_scores(self, destroy_op: str, repair_op: str, score: float) -> None:
        """
        Update operator scores for the current segment.

        Args:
            destroy_op: Name of the destroy operator used.
            repair_op: Name of the repair operator used.
            score: Performance score (higher is better).
        """
        self.destroy_scores[destroy_op] += score
        self.repair_scores[repair_op] += score
        self.destroy_counts[destroy_op] += 1
        self.repair_counts[repair_op] += 1

        self.current_iteration += 1
        if self.current_iteration >= self.segment_size:
            self._end_segment()
            self.current_iteration = 0

    def _end_segment(self) -> None:
        """Update weights at the end of a segment using the reaction factor."""
        for op in self.destroy_weights:
            if self.destroy_counts[op] > 0:
                avg_score = self.destroy_scores[op] / self.destroy_counts[op]
                self.destroy_weights[op] = (
                    self.reaction_factor * avg_score + (1 - self.reaction_factor) * self.destroy_weights[op]
                )
            self.destroy_scores[op] = 0.0
            self.destroy_counts[op] = 0

        for op in self.repair_weights:
            if self.repair_counts[op] > 0:
                avg_score = self.repair_scores[op] / self.repair_counts[op]
                self.repair_weights[op] = (
                    self.reaction_factor * avg_score + (1 - self.reaction_factor) * self.repair_weights[op]
                )
            self.repair_scores[op] = 0.0
            self.repair_counts[op] = 0

    def decay_weights(self) -> None:
        """Deprecated in favor of segment-based updates, but kept for compatibility."""
        pass

    def entropy(self) -> float:
        """
        Compute entropy of operator weight distributions (measure of diversity).

        Returns:
            Combined entropy of destroy and repair weights.
        """
        destroy_entropy = self._compute_entropy(list(self.destroy_weights.values()))
        repair_entropy = self._compute_entropy(list(self.repair_weights.values()))
        return destroy_entropy + repair_entropy

    @staticmethod
    def _compute_entropy(weights: List[float]) -> float:
        """Compute Shannon entropy of a probability distribution."""
        total = sum(weights)
        if total == 0:
            return 0.0
        probs = [w / total for w in weights]
        return -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

    @staticmethod
    def _roulette_wheel(weights: Dict[str, float], rng: random.Random) -> str:
        """Roulette wheel selection based on weights."""
        total = sum(weights.values())
        if total == 0:
            return rng.choice(list(weights.keys()))

        r = rng.uniform(0, total)
        cumulative = 0.0
        for op, weight in weights.items():
            cumulative += weight
            if cumulative >= r:
                return op

        return list(weights.keys())[-1]  # Fallback


class RuinRecreateOperator:
    """
    Applies destroy and repair operators to an Individual.

    This class integrates ALNS-style ruin-and-recreate with the HGS framework,
    converting between giant tour representation and route-based representation.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        params: HGSRRParams,
        split_manager,
    ):
        """
        Initialize the ruin-recreate operator.

        Args:
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
            params: HGSRR parameters.
            split_manager: Split algorithm manager for evaluating giant tours.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit
        self.params = params
        self.split_manager = split_manager
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()

    def apply(
        self,
        individual: Individual,
        destroy_operator: str,
        repair_operator: str,
    ) -> Individual:
        """
        Apply ruin-and-recreate to an individual.

        Args:
            individual: The individual to mutate.
            destroy_operator: Name of the destroy operator to use.
            repair_operator: Name of the repair operator to use.

        Returns:
            New mutated individual.
        """
        # 1. Convert giant tour to routes
        evaluate(individual, self.split_manager)
        routes = [route[:] for route in individual.routes]  # Deep copy

        if not routes or all(len(r) == 0 for r in routes):
            return individual  # Nothing to destroy

        # 2. Determine removal size
        total_nodes = sum(len(r) for r in routes)
        if total_nodes == 0:
            return individual

        min_remove = max(1, int(total_nodes * self.params.min_removal_pct))
        max_remove = max(min_remove, int(total_nodes * self.params.max_removal_pct))
        n_remove = self.rng.randint(min_remove, max_remove)

        # 3. Apply destroy operator
        destroyed_routes, removed_nodes = self._apply_destroy(routes, destroy_operator, n_remove)

        # 4. Apply repair operator
        repaired_routes = self._apply_repair(destroyed_routes, removed_nodes, repair_operator)

        # 5. Convert routes back to giant tour
        new_giant_tour = self._routes_to_giant_tour(repaired_routes)

        # 6. Ensure the giant tour contains ALL nodes (permutation property)
        # Any nodes not selected for routes are appended at the end
        all_nodes_set = set(range(1, len(self.dist_matrix)))
        visited_set = set(new_giant_tour)
        missing_nodes = sorted(list(all_nodes_set - visited_set))
        new_giant_tour.extend(missing_nodes)

        if len(new_giant_tour) == 0:
            return individual

        # 7. Create and evaluate new individual
        new_individual = Individual(new_giant_tour)
        evaluate(new_individual, self.split_manager)

        return new_individual

    def _apply_destroy(
        self, routes: List[List[int]], operator_name: str, n_remove: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Apply a destroy operator to the routes.

        Args:
            routes: List of routes.
            operator_name: Name of the destroy operator.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple of (modified routes, list of removed node IDs).
        """
        # Map operator name to function
        operator_map = {
            "random_removal": destroy_operators.random_removal,
            "worst_removal": (
                destroy_operators.worst_profit_removal
                if getattr(self.params, "profit_aware_operators", False)
                else destroy_operators.worst_removal
            ),
            "cluster_removal": destroy_operators.cluster_removal,
            "shaw_removal": (
                destroy_operators.shaw_profit_removal
                if getattr(self.params, "profit_aware_operators", False)
                else destroy_operators.shaw_removal
            ),
            "string_removal": destroy_operators.string_removal,
        }

        operator_func = operator_map.get(operator_name)
        if operator_func is None:
            # Fallback to random removal
            operator_func = destroy_operators.random_removal

        all_nodes = list(range(1, len(self.dist_matrix)))

        # Call the operator
        try:
            # We use kwargs to match different operator signatures
            kwargs = {
                "routes": routes,
                "n_remove": n_remove,
                "dist_matrix": self.dist_matrix,
                "rng": self.rng,
            }

            # Add operator-specific arguments
            if operator_name in ["shaw_removal", "cluster_removal"]:
                kwargs["nodes"] = all_nodes

            # Add profit-aware arguments if needed
            if getattr(self.params, "profit_aware_operators", False):
                kwargs["wastes"] = self.wastes
                kwargs["R"] = self.revenue
                kwargs["C"] = self.cost_unit

            elif "profit" in operator_name or operator_name in ["shaw_removal", "worst_removal"]:
                kwargs["wastes"] = self.wastes

            # Clean kwargs based on actual function inspection or just use a safer approach
            # Using common argument names used in the package
            modified_routes, removed = operator_func(**kwargs)  # type: ignore[arg-type]
            return modified_routes, removed
        except Exception:
            # Fallback: try positional call for very basic ones if kwargs fail
            try:
                modified_routes, removed = operator_func(routes, n_remove, self.dist_matrix)  # type: ignore[arg-type,call-arg]
                return modified_routes, removed
            except Exception:
                # Absolute fallback: return original routes and empty removed list
                return routes, []

    def _apply_repair(self, routes: List[List[int]], removed_nodes: List[int], operator_name: str) -> List[List[int]]:
        """
        Apply a repair operator to reinsert removed nodes.

        Args:
            routes: Current (partial) routes.
            removed_nodes: List of node IDs to reinsert.
            operator_name: Name of the repair operator.

        Returns:
            Repaired routes.
        """
        if not removed_nodes:
            return routes

        # Map operator name to function
        operator_map = {
            "greedy_insertion": (
                repair_operators.greedy_profit_insertion
                if getattr(self.params, "profit_aware_operators", False)
                else repair_operators.greedy_insertion
            ),
            "regret_2_insertion": (
                repair_operators.regret_2_profit_insertion
                if getattr(self.params, "profit_aware_operators", False)
                else repair_operators.regret_2_insertion
            ),
            "regret_k_insertion": repair_operators.regret_k_insertion,
            "greedy_insertion_with_blinks": (
                repair_operators.greedy_profit_insertion_with_blinks
                if getattr(self.params, "profit_aware_operators", False)
                else repair_operators.greedy_insertion_with_blinks
            ),
        }

        operator_func = operator_map.get(operator_name)
        if operator_func is None:
            operator_func = repair_operators.greedy_insertion

        # Call the operator
        try:
            # Common parameters for all operators
            kwargs = {
                "routes": routes,
                "removed_nodes": removed_nodes,
                "dist_matrix": self.dist_matrix,
                "wastes": self.wastes,
                "capacity": self.capacity,
            }

            # Map specific parameters based on operator type
            kwargs["mandatory_nodes"] = list(self.split_manager.mandatory_nodes)
            kwargs["expand_pool"] = getattr(self.params, "vrpp", True)
            if getattr(self.params, "profit_aware_operators", False) and operator_name != "regret_k_insertion":
                kwargs["R"] = self.revenue
                kwargs["C"] = self.cost_unit
            elif "blink" in operator_name:
                kwargs["blink_rate"] = self.params.noise_factor
                kwargs["rng"] = self.rng

            repaired_routes = operator_func(**kwargs)  # type: ignore[arg-type]
            return repaired_routes
        except Exception:
            # Fallback: append uninserted nodes as new routes if capacity allows
            for node in removed_nodes:
                demand = self.wastes.get(node, 0)
                if demand <= self.capacity:
                    routes.append([node])
            return routes

    @staticmethod
    def _routes_to_giant_tour(routes: List[List[int]]) -> List[int]:
        """
        Convert routes to a giant tour (concatenated node sequence).

        Args:
            routes: List of routes.

        Returns:
            Flattened list of node IDs.
        """
        giant_tour = []
        for route in routes:
            giant_tour.extend(route)
        return giant_tour
