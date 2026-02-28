"""
Solution Construction Module for K-Sparse ACO in Build Optimization.

Ants construct solutions by probabilistically selecting items for each slot based on
pheromone levels and heuristic information (e.g. cost efficiency).
"""

import random

import numpy as np

from core.problem import BuildProblem

from .params import ACOParams
from .pheromones import SparsePheromoneTau


class SolutionConstructor:
    """
    Constructs a single build solution for an ant using the
    sparse pheromone matrix and heuristic values.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        pheromone: SparsePheromoneTau,
        eta: np.ndarray,
        params: ACOParams,
        tau_0: float,
    ):
        """
        Initialize Solution Constructor.

        Args:
            problem: BuildProblem instance with items and costs.
            budget: Maximum allowed cost.
            pheromone: Sparse pheromone matrix.
            eta: Heuristic information array (e.g., efficiency per item).
            params: ACO parameters.
            tau_0: Initial pheromone value.
        """
        self.problem = problem
        self.budget = budget
        self.pheromone = pheromone
        self.eta = eta
        self.params = params
        self.tau_0 = tau_0

    def construct(self) -> np.ndarray:
        """
        Construct a solution using the ACS state transition rule.

        Returns:
            np.ndarray: Array of selected item indices of length num_slots.
        """
        num_slots = self.problem.num_slots

        build = np.full(num_slots, -1, dtype=int)
        current_cost = 0.0

        for slot_idx in range(num_slots):
            # Find feasible items that don't exceed budget
            remaining_budget = self.budget - current_cost
            feasible = np.where(self.problem.costs <= remaining_budget)[0]

            if len(feasible) == 0:
                # If no items are feasible for this slot, we are stuck.
                # Since slots might be mandatory, we just pick the cheapest item
                # to minimally violate constraint, or random.
                feasible = np.array([np.argmin(self.problem.costs)])

            # Select next item
            selected_item = self._select_next_item(slot_idx, feasible)

            # Local pheromone update (ACS rule)
            self._local_pheromone_update(slot_idx, selected_item)

            build[slot_idx] = selected_item
            current_cost += self.problem.costs[selected_item]

        return build

    def _select_next_item(self, slot_idx: int, feasible: np.ndarray) -> int:
        """
        Select next item using pseudo-random proportional rule.
        """
        if random.random() < self.params.q0:
            # Exploitation: select best
            best_val = -1.0
            best_item = feasible[0]
            for item_idx in feasible:
                tau = self.pheromone.get(slot_idx, item_idx)
                eta = self.eta[item_idx]
                val = (tau**self.params.alpha) * (eta**self.params.beta)
                if val > best_val:
                    best_val = val
                    best_item = item_idx
            return int(best_item)
        else:
            # Exploration: proportional selection
            probs = []
            for item_idx in feasible:
                tau = self.pheromone.get(slot_idx, item_idx)
                eta = self.eta[item_idx]
                probs.append((tau**self.params.alpha) * (eta**self.params.beta))

            total = sum(probs)
            if total <= 0:
                return int(random.choice(feasible))

            r = random.uniform(0, total)
            cumsum = 0.0
            for idx, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    return int(feasible[idx])

            return int(feasible[-1])

    def _local_pheromone_update(self, slot_idx: int, item_idx: int) -> None:
        """
        Apply ACS local pheromone update rule.

        tau(s, i) = (1 - rho) * tau(s, i) + rho * tau_0
        """
        rho = self.params.rho
        current = self.pheromone.get(slot_idx, item_idx)
        new_value = (1 - rho) * current + rho * self.tau_0
        self.pheromone.set(slot_idx, item_idx, new_value)
