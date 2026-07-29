"""
Internal implementation of Iterative 2-opt for TSP.
"""

from typing import List

import numpy as np


def solve_tsp_2opt(dist_matrix: np.ndarray, nodes: List[int], depot: int = 0, max_iterations: int = 1000) -> List[int]:
    """
    Solve TSP via iterative 2-opt.
    """
    if not nodes:
        return [depot]

    # Initial tour: depot -> nodes -> depot
    tour = [depot] + list(nodes) + [depot]

    n = len(tour)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Current edges: (i-1, i) and (j, j+1)
                # Potential edges: (i-1, j) and (i, j+1)
                d_curr = dist_matrix[tour[i - 1], tour[i]] + dist_matrix[tour[j], tour[j + 1]]
                d_new = dist_matrix[tour[i - 1], tour[j]] + dist_matrix[tour[i], tour[j + 1]]

                if d_new < d_curr - 1e-4:
                    # Reverse segment [i:j+1]
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    improved = True
        iteration += 1

    return tour
