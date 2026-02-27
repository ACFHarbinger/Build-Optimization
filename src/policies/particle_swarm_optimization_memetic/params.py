"""
Configuration parameters for the Particle Swarm Optimization Memetic Algorithm (PSOMA).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PSOMAParams:
    """
    Configuration parameters for the PSOMA solver.

    A particle's position is a set of routes (permutation).  Velocity is
    implemented as a list of (node_swap | OX-segment) operations applied
    probabilistically from the personal best (pbest) and global best (gbest).

    Attributes:
        pop_size: Swarm size (number of particles).
        omega: Inertia weight — fraction of current velocity retained.
        c1: Cognitive acceleration coefficient (toward pbest).
        c2: Social acceleration coefficient (toward gbest).
        max_iterations: Maximum PSO iterations.
        local_search_freq: Every N iterations, apply local search to all particles.
        n_removal: Number of nodes removed per local-search step.
        time_limit: Wall-clock time limit in seconds.
    """

    pop_size: int = 20
    omega: float = 0.1
    c1: float = 1.5
    c2: float = 2.0
    max_iterations: int = 200
    local_search_freq: int = 10
    n_removal: int = 2
    time_limit: float = 60.0
