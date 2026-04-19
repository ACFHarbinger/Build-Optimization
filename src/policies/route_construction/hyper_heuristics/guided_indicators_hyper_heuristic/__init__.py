"""
GIHH (Hyper-Heuristic with Two Guidance Indicators) policy module.

This module implements a selection hyper-heuristic that uses two guidance
indicators to adaptively select low-level heuristics during search:
1. Improvement Rate Indicator (IRI): Measures solution quality improvement
2. Time-based Indicator (TBI): Measures computational efficiency

Reference:
    Kheiri, A., & Keedwell, E. (2015). A sequence-based selection hyper-heuristic
    utilising a hidden Markov model. In Proceedings of the 2015 Annual Conference
    on Genetic and Evolutionary Computation (pp. 417-424).
"""

from .gihh import GIHHSolver
from .params import GIHHParams

__all__ = ["GIHHSolver", "GIHHParams"]
