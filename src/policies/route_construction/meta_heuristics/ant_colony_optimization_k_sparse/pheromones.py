"""
Sparse Pheromone Management Module.

This module implements the sparse pheromone matrix used in K-Sparse ACO
following the MMAS_exp methodology from Hale (2021).

It uses dynamic default_value tracking and precision-based pruning
instead of fixed-capacity storage.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse.pheromones import SparsePheromoneTau
    >>> pheromone = SparsePheromoneTau(n_nodes=100, tau_0=1.0, scale=5.0, ...)
    >>> val = pheromone.get(0, 1)

Reference:
    Hale, D. "Investigation of Ant Colony Optimization Implementation
    Strategies For Low-Memory Operating Environments", 2021.
    Section 4.2.3: MMAS_exp with scale-based precision pruning.
"""

from collections import defaultdict
from typing import Dict


class SparsePheromoneTau:
    """
    Sparse pheromone matrix using MMAS_exp with precision-based pruning.

    Instead of maintaining a fixed k-capacity cache, this implementation
    tracks a dynamic default_value that evaporates globally. Edge values
    are only stored explicitly if they differ from default_value by more
    than a precision threshold (10^-scale).

    This approach follows the experimental MAX-MIN Ant System (MMAS_exp)
    described in Hale (2021), optimizing both memory usage and computation
    time while maintaining MMAS convergence properties.
    """

    def __init__(self, n_nodes: int, tau_0: float, scale: float, tau_min: float, tau_max: float):
        """
        Initialize sparse pheromone structure with scale-based pruning.

        Args:
            n_nodes: Total number of nodes (including depot).
            tau_0: Initial pheromone value (becomes initial default_value).
            scale: Precision parameter. Values within 10^-scale of default_value are pruned.
            tau_min: Minimum pheromone bound (MMAS lower limit).
            tau_max: Maximum pheromone bound (MMAS upper limit).
        """
        self.n_nodes = n_nodes
        self.scale = scale
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Dynamic default value that evaporates globally
        self.default_value = tau_0

        # Sparse storage: node -> {neighbor: pheromone}
        # Only stores edges that differ significantly from default_value
        self._pheromone: Dict[int, Dict[int, float]] = defaultdict(dict)

    def get(self, i: int, j: int) -> float:
        """
        Get pheromone value for edge (i, j).

        Args:
            i: Source node index.
            j: Destination node index.

        Returns:
            Pheromone value (explicit if stored, otherwise default_value).
        """
        if j in self._pheromone[i]:
            return self._pheromone[i][j]
        return self.default_value

    def set(self, i: int, j: int, value: float) -> None:
        """
        Set pheromone value with MMAS bounds.

        This method does NOT enforce capacity limits. Instead, precision-based
        pruning during evaporation controls memory usage.

        Args:
            i: Source node index.
            j: Destination node index.
            value: New pheromone value (will be clamped to [tau_min, tau_max]).
        """
        # Apply MMAS bounds
        value = max(self.tau_min, min(self.tau_max, value))

        # Store the value explicitly (pruning happens during evaporation)
        self._pheromone[i][j] = value

    def deposit_edge(self, i: int, j: int, delta: float) -> None:
        """
        Deposit pheromone on edge (i, j).

        Adds delta to the current pheromone value. Used during global
        pheromone updates in MMAS.

        Args:
            i: Source node index.
            j: Destination node index.
            delta: Amount of pheromone to deposit.
        """
        current = self.get(i, j)
        self.set(i, j, current + delta)

    def evaporate_all(self, rho: float) -> None:
        """
        Apply MMAS_exp global evaporation with precision-based pruning.

        This method implements the core memory optimization from Hale (2021):
        1. Evaporate the default_value globally
        2. Evaporate all explicitly stored edges
        3. Prune edges that are within precision tolerance of default_value

        Args:
            rho: Evaporation rate (0 < rho < 1).
        """
        # Step 1: Evaporate the default value and apply MMAS lower bound
        self.default_value = max(self.tau_min, self.default_value * (1 - rho))

        # Step 2: Calculate precision threshold for pruning
        precision = 10**-self.scale

        # Step 3: Evaporate all explicit edges and prune if close to default
        for i in list(self._pheromone.keys()):
            for j in list(self._pheromone[i].keys()):
                # Evaporate the stored value
                self._pheromone[i][j] *= 1 - rho

                # Apply MMAS bounds
                self._pheromone[i][j] = max(self.tau_min, min(self.tau_max, self._pheromone[i][j]))

                # Precision pruning: delete if within tolerance of default_value
                if abs(self._pheromone[i][j] - self.default_value) <= precision:
                    del self._pheromone[i][j]

            # Clean up empty dictionaries to free memory
            if not self._pheromone[i]:
                del self._pheromone[i]
