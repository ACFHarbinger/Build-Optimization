"""
Configuration parameters for the Genetic Programming Hyper-Heuristic (GPHH) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPHHParams:
    """
    Configuration parameters for the GPHH solver.

    GPHH evolves selection policies (GP trees) that govern which Low-Level
    Heuristic (LLH) to apply at each step, rather than evolving routing
    solutions directly.  The best policy tree is then applied for a longer
    run to produce the final solution.

    LLH Pool (5 operators, indices 0-4):
      L0: random_removal  + greedy_insertion
      L1: worst_removal   + regret_2_insertion
      L2: cluster_removal + greedy_insertion
      L3: worst_removal   + greedy_insertion
      L4: random_removal  + regret_2_insertion

    GP Trees are binary expression trees of depth ≤ `tree_depth`.
    Terminal nodes yield feature scalars; function nodes combine them.

    Attributes:
        gp_pop_size: Number of GP trees in the meta-population.
        max_gp_generations: GP evolution generations.
        eval_steps: LLH applications per tree fitness evaluation.
        apply_steps: LLH applications when running the best tree at the end.
        tree_depth: Maximum depth of GP expression trees.
        tournament_size: Tournament size for GP selection.
        n_llh: Number of Low-Level Heuristics in the pool.
        n_removal: Nodes removed per LLH destroy step.
        time_limit: Wall-clock time limit in seconds.
    """

    gp_pop_size: int = 20
    max_gp_generations: int = 30
    eval_steps: int = 50
    apply_steps: int = 200
    tree_depth: int = 3
    tournament_size: int = 3
    n_llh: int = 5
    n_removal: int = 2
    time_limit: float = 60.0
