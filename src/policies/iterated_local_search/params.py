"""
Configuration parameters for the Iterated Local Search (ILS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ILSParams:
    """
    Configuration for the ILS solver.

    ILS alternates between a local search descent phase and a perturbation
    phase that escapes local optima.  Acceptance is based on strict profit
    improvement (hill-climbing) with the perturbation providing diversification.

    Attributes:
        n_restarts: Number of perturbation + descent cycles.
        inner_iterations: LLH iterations per descent phase.
        n_removal: Nodes removed per LLH destroy step.
        perturbation_strength: Fraction of nodes perturbed.
        time_limit: Wall-clock time limit in seconds.
    """

    n_restarts: int = 50
    inner_iterations: int = 100
    n_removal: int = 2
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
