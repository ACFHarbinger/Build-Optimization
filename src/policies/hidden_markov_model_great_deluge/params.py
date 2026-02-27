"""
Configuration parameters for the HMM + Great Deluge (HMM-GD) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HMMGDParams:
    """
    Configuration parameters for the HMM-GD solver.

    The solver is an online-learning hyper-heuristic.  A Hidden Markov Model
    (HMM) learns which Low-Level Heuristic (LLH) to invoke based on the
    observed sequence of search states (improving / stagnating / escaping).
    The Great Deluge criterion replaces simulated annealing: solutions are
    accepted when their profit exceeds a monotonically falling water level.

    LLH Pool (5 operators, indices 0-4):
      L0: random_removal  + greedy_insertion
      L1: worst_removal   + regret_2_insertion
      L2: cluster_removal + greedy_insertion
      L3: worst_removal   + greedy_insertion
      L4: random_removal  + regret_2_insertion

    HMM States: 0=improving, 1=stagnating, 2=escaping

    Attributes:
        max_iterations: Total LLH applications.
        flood_margin: Initial water level = best_profit * (1 + flood_margin).
        rain_speed: Water level decrease per iteration (absolute profit units).
        learning_rate: Online HMM transition probability update step size.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool (fixed at 5).
        time_limit: Wall-clock time limit in seconds.
    """

    max_iterations: int = 500
    flood_margin: float = 0.05
    rain_speed: float = 0.001
    learning_rate: float = 0.1
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
