"""
Configuration parameters for the Genetic Programming Hyper-Heuristic (GP-HH).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GPHHParams:
    """
    Configuration for the Genetic Programming Hyper-Heuristic solver.

    **Generative Architecture** (Burke et al., 2009):
    The GP tree evaluates candidate (node, route, position) insertions and
    returns a utility score.  The constructive algorithm greedily executes
    the highest-scored, capacity-feasible insertion at each step.

    **K-NN Candidate List**:
    Rather than scoring all N unvisited nodes per route, only the
    ``candidate_list_size`` nearest neighbours of the route endpoints are
    evaluated.  Reduces construction cost from O(N² R) to O(N K R).

    **Train/Test Paradigm**:
    During GP evolution, each tree's fitness is its average normalised profit
    across training environments (fully distinct distance matrices if supplied
    by the adapter, otherwise node-subset fallback on the test instance).

    Attributes:
        gp_pop_size: Number of GP trees in the meta-population.
        max_gp_generations: GP evolution generations.
        tree_depth: Maximum depth of GP expression trees.
        tournament_size: Tournament size for GP selection.
        time_limit: Wall-clock time limit in seconds.
        parsimony_coefficient: Weight for tree-size penalty (0.0 = none).
        candidate_list_size: K in K-NN candidate filtering (≥ 1).
        n_training_instances: Sub-instances for fitness averaging (fallback).
        training_sample_ratio: Fraction of nodes sampled per fallback instance.
        seed: Random seed for reproducibility.
        vrpp: If True, nodes are optional (may be skipped for profitability).
    """

    gp_pop_size: int = 20
    max_gp_generations: int = 30
    tree_depth: int = 3
    tournament_size: int = 3
    time_limit: float = 60.0
    parsimony_coefficient: float = 0.001
    candidate_list_size: int = 10
    n_training_instances: int = 3
    training_sample_ratio: float = 0.5

    # Infrastructure
    seed: Optional[int] = None
    vrpp: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "GPHHParams":
        """Create parameters from a configuration object."""
        return cls(
            gp_pop_size=getattr(config, "gp_pop_size", 20),
            max_gp_generations=getattr(config, "max_gp_generations", 30),
            tree_depth=getattr(config, "tree_depth", 3),
            tournament_size=getattr(config, "tournament_size", 3),
            time_limit=getattr(config, "time_limit", 60.0),
            parsimony_coefficient=getattr(config, "parsimony_coefficient", 0.001),
            candidate_list_size=getattr(config, "candidate_list_size", 10),
            n_training_instances=getattr(config, "n_training_instances", 3),
            training_sample_ratio=getattr(config, "training_sample_ratio", 0.5),
            vrpp=getattr(config, "vrpp", True),
        )
