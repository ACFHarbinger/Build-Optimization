"""
K-Sparse Ant Colony Optimization.

This package implements the K-Sparse ACO algorithm, which restricts the
search space to the k-nearest neighbors for efficiency.

Attributes:
    KSparseACOSolver (class): The main solver class.
    run_k_sparse_aco (function): Helper function to run the solver.

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse import run_k_sparse_aco
    >>> result = run_k_sparse_aco(dist_matrix, wastes, ...)
"""

from .solver import KSparseACOSolver

__all__ = ["KSparseACOSolver"]
