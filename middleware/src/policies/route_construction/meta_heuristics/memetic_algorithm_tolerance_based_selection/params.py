"""
Configuration parameters for Stochastic Tournament Genetic Algorithm (STGA).

This is the rigorous parameter mapping for the League Championship Algorithm (LCA)
with proper Operations Research terminology.

TERMINOLOGY MAPPING (LCA → MA-TS):
- n_teams → population_size
- tolerance_pct → tolerance_pct (UNCHANGED - critical LCA feature)
- crossover_prob → recombination_rate
- n_removal → perturbation_strength
- max_iterations → max_iterations

Reference:
    Kashan, A. H. (2013). "League Championship Algorithm (LCA): An algorithm
    for global optimization inspired by sport championships."
    Applied Soft Computing, 13(5), 2171-2200.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemeticAlgorithmToleranceBasedSelectionParams:
    """
    Parameters for Memetic Algorithm with Tolerance-based Selection (MA-TS).

    Algorithm Structure:
        1. Initialize population of N solutions
        2. For each generation:
            a. Round-Robin Schedule: Random pairwise matching
            b. For each match (i vs j):
                - Determine winner using infeasibility tolerance
                - If |fitness_i - fitness_j| ≤ tolerance: random winner
                - Otherwise: higher fitness wins
                - Loser generates new solution (recombination or mutation)
                - Loser ALWAYS accepts new solution

    KEY FEATURE: Infeasibility Tolerance
        Solutions with similar fitness compete randomly, preserving diversity
        and allowing exploration of alternative feasible basins. This is the
        distinguishing characteristic of LCA vs standard tournament GA.

    Attributes:
        population_size: Number of candidate solutions (LCA: n_teams).
        max_iterations: Maximum number of evolution cycles (LCA: max_iterations).
        tolerance_pct: Infeasibility tolerance as fraction of average fitness.
                      Solutions within this tolerance compete randomly.
                      (LCA: tolerance_pct). Typical: 0.01-0.10.
        recombination_rate: Probability of crossover vs mutation (LCA: crossover_prob).
        perturbation_strength: Number of nodes to remove in mutation (LCA: n_removal).
        local_search_iterations: Local search refinement iterations.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Mathematical Foundation:
        Infeasibility Tolerance Formula:
            tolerance = tolerance_pct × (|fitness_a| + |fitness_b|) / 2
            if |fitness_a - fitness_b| ≤ tolerance:
                winner = random choice (diversity preservation)
            else:
                winner = argmax(fitness_a, fitness_b)

    Complexity:
        - Space: O(N × n) for population storage
        - Time per generation: O(N × eval_cost) for pairwise matches
    """

    seed: Optional[int] = None

    # Population structure (LCA: n_teams)
    population_size: int = 10

    # Evolution control (LCA: max_iterations)
    max_iterations: int = 100

    # Infeasibility tolerance (LCA: tolerance_pct)
    # CRITICAL FEATURE: Allows diversity preservation
    tolerance_pct: float = 0.05  # 5% tolerance for similar solutions

    # Genetic operators (LCA: crossover_prob, n_removal)
    recombination_rate: float = 0.6  # Probability of crossover vs mutation
    perturbation_strength: int = 2  # Nodes removed in mutation
    n_removal: int = 1

    # Local search refinement
    local_search_iterations: int = 100

    # Profit-awareness flags
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Resource constraints
    time_limit: float = 60.0

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmToleranceBasedSelectionParams":
        """Create parameters from a configuration object."""
        return cls(
            population_size=getattr(config, "population_size", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            tolerance_pct=getattr(config, "tolerance_pct", 0.05),
            recombination_rate=getattr(config, "recombination_rate", 0.6),
            perturbation_strength=getattr(config, "perturbation_strength", 2),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
        )

    def __post_init__(self):
        """Validate parameter constraints."""
        assert self.population_size > 0, "population_size must be positive"
        assert self.population_size % 2 == 0, "population_size must be even for pairwise matching"
        assert 0 <= self.tolerance_pct <= 1, "tolerance_pct must be in [0, 1]"
        assert 0 <= self.recombination_rate <= 1, "recombination_rate must be in [0, 1]"
        assert self.perturbation_strength > 0, "perturbation_strength must be positive"
        assert self.n_removal > 0, "n_removal must be positive"
