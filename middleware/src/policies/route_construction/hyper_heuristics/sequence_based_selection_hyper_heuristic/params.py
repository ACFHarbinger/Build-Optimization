"""
Configuration parameters for the Sequence-based Selection Hyper-Heuristic (SS-HH).

Reference:
    Kheiri, A. "Heuristic Sequence Selection for Inventory Routing Problem", 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SSHHParams:
    """
    Configuration parameters for the SS-HH solver (Algorithm 1).

    The solver builds *sequences* of Low-Level Heuristics (LLHs) driven by
    two HMM-style score matrices:

    -  TMatrix[h_prev][h_cur]: transition scores between LLHs (Eq. 2).
    -  ASMatrix[h_cur][AS]:  acceptance-strategy scores; AS=0 extends the
       sequence, AS=1 triggers application (Eq. 3).

    All matrix entries are initialised to 1 and incremented by Δ_norm
    whenever a sequence improves the global best solution.

    LLH Pool (5 operators, indices 0-4):
      L0: random_removal  + greedy_insertion
      L1: worst_removal   + regret_2_insertion
      L2: cluster_removal + greedy_insertion
      L3: worst_removal   + greedy_insertion
      L4: random_removal  + regret_2_insertion

    Attributes:
        max_iterations: Total main-loop steps (each step either extends or
            applies the current heuristic sequence).
        n_removal: Nodes removed per LLH destroy step.
        n_llh: Number of LLHs in the pool (fixed at 5).
        time_limit: Wall-clock time limit in seconds.
        threshold_infeasible: Acceptance threshold T when the best solution
            is infeasible (Eq. 4, top branch).
        threshold_feasible_base: Base acceptance threshold T for feasible
            solutions (Eq. 4, bottom branch, first addend).
        threshold_decay_rate: Time-decay coefficient for the feasible
            threshold (Eq. 4, bottom branch, second addend).
    """

    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    threshold_infeasible: float = 0.001
    threshold_feasible_base: float = 0.0001
    threshold_decay_rate: float = 0.01

    # Profit-awareness
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "SSHHParams":
        """Create parameters from a configuration object."""
        return cls(
            max_iterations=getattr(config, "max_iterations", 500),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            time_limit=getattr(config, "time_limit", 60.0),
            threshold_infeasible=getattr(config, "threshold_infeasible", 0.001),
            threshold_feasible_base=getattr(config, "threshold_feasible_base", 0.0001),
            threshold_decay_rate=getattr(config, "threshold_decay_rate", 0.01),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
