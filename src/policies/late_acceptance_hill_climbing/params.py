"""
Configuration parameters for the Late Acceptance Hill-Climbing (LAHC) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LAHCParams:
    """
    Configuration parameters for the LAHC solver.

    LAHC compares a candidate solution's objective not against the current
    solution but against the solution from ``queue_size`` iterations ago,
    stored in a circular queue.  This deferred comparison induces a dynamic
    cooling effect without explicit temperature scheduling.

    LLH Pool (5 operators, indices 0-4):
      L0: random_removal  + greedy_insertion
      L1: worst_removal   + regret_2_insertion
      L2: cluster_removal + greedy_insertion
      L3: worst_removal   + greedy_insertion
      L4: random_removal  + regret_2_insertion

    Attributes:
        queue_size: Length of the circular history queue (L).
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per LLH destroy step.
        n_llh: Number of Low-Level Heuristics in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    queue_size: int = 50
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
