"""
Configuration parameters for the Variable Neighborhood Search (VNS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VNSParams:
    """
    Configuration for the VNS solver.

    Systematically explores a hierarchy of shaking neighborhoods (N_1 ... N_{k_max})
    with a local search descent between each shaking step.  An improvement resets
    k to 1; exhausting all k_max structures completes one outer iteration.

    Attributes:
        k_max: Number of shaking neighborhood structures (N_1 ... N_{k_max}).
        max_iterations: Total outer VNS iterations.
        local_search_iterations: LLH attempts per local search descent phase.
        n_removal: Nodes removed per LLH destroy step in local search.
        n_llh: Number of LLHs in the local search pool.
        time_limit: Wall-clock time limit in seconds.
    """

    k_max: int = 5
    max_iterations: int = 200
    local_search_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
