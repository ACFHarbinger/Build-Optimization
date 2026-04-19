"""
Parameters for Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR).

This module defines the configuration parameters for the HGS-RR algorithm,
extending standard HGS with adaptive destroy/repair operator management.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HGSRRParams:
    """
    Configuration parameters for HGS-RR algorithm.

    Attributes:
        restart_timer (float): Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit (float): Maximum execution time in seconds.
        mu (int): Minimum population size (for each subpopulation).
        nb_elite (int): Number of elite individuals preserved.
        mutation_rate (float): Probability of applying ruin-recreate mutation.
        n_offspring (int): Generation size (individuals created per iteration).
        n_iterations_no_improvement (int): Max iterations without improvement.
        survivor_threshold (float): Population size multiplier for survivor selection.
        max_vehicles (int): Maximum number of vehicles (0 = unlimited).
        crossover_rate (float): Probability of crossover operation.
        nb_granular (int): Granular search parameter for local search.

        # Ruin-and-Recreate specific parameters
        min_removal_pct (float): Minimum percentage of nodes to remove (0.0-1.0).
        max_removal_pct (float): Maximum percentage of nodes to remove (0.0-1.0).
        noise_factor (float): Noise factor for randomized best insertion.
        reaction_factor (float): Rate of operator weight updates.
        decay_parameter (float): Decay rate for operator scores.

        # Operator selection weights (initial)
        destroy_operators (List[str]): List of destroy operator names.
        repair_operators (List[str]): List of repair operator names.
        operator_decay_rate (float): Exponential decay rate for operator weights.

        # Scoring parameters for adaptive selection
        score_sigma_1 (float): Score for new global best solution.
        score_sigma_2 (float): Score for solution improving current.
        score_sigma_3 (float): Score for solution accepted but not improving.

        seed (Optional[int]): Random seed for reproducibility.
    """

    # Core HGS parameters
    restart_timer: float = 0.0
    time_limit: float = 10.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.3
    n_iterations_no_improvement: int = 20000
    no_improvement_threshold: int = 20  # Threshold for diversity/stopping
    survivor_threshold: float = 2.0
    max_vehicles: int = 0
    crossover_rate: float = 0.7
    neighbor_list_size: int = 10

    # Ruin-and-Recreate parameters
    min_removal_pct: float = 0.1
    max_removal_pct: float = 0.4
    noise_factor: float = 0.015
    reaction_factor: float = 0.1
    decay_parameter: float = 0.95

    # Operator management
    destroy_operators: List[str] = field(
        default_factory=lambda: [
            "random_removal",
            "worst_removal",
            "cluster_removal",
            "shaw_removal",
            "string_removal",
        ]
    )
    repair_operators: List[str] = field(
        default_factory=lambda: [
            "greedy_insertion",
            "regret_2_insertion",
            "regret_k_insertion",
            "greedy_insertion_with_blinks",
        ]
    )
    operator_decay_rate: float = 0.95

    # Adaptive scoring
    score_sigma_1: float = 33.0  # New global best
    score_sigma_2: float = 9.0  # Improvement
    score_sigma_3: float = 3.0  # Accepted

    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "HGSRRParams":
        """Create parameters from a configuration object."""
        return cls(
            time_limit=getattr(config, "time_limit", 10.0),
            population_size=getattr(config, "population_size", 50),
            elite_size=getattr(config, "elite_size", 10),
            mutation_rate=getattr(config, "mutation_rate", 0.3),
            n_iterations_no_improvement=getattr(config, "n_iterations_no_improvement", 20000),
            no_improvement_threshold=getattr(config, "no_improvement_threshold", 20),
            survivor_threshold=getattr(config, "survivor_threshold", 2.0),
            max_vehicles=getattr(config, "max_vehicles", 0),
            crossover_rate=getattr(config, "crossover_rate", 0.7),
            neighbor_list_size=getattr(config, "neighbor_list_size", 10),
            min_removal_pct=getattr(config, "min_removal_pct", 0.1),
            max_removal_pct=getattr(config, "max_removal_pct", 0.4),
            noise_factor=getattr(config, "noise_factor", 0.015),
            reaction_factor=getattr(config, "reaction_factor", 0.1),
            decay_parameter=getattr(config, "decay_parameter", 0.95),
            destroy_operators=getattr(
                config,
                "destroy_operators",
                ["random_removal", "worst_removal", "cluster_removal", "shaw_removal", "string_removal"],
            ),
            repair_operators=getattr(
                config,
                "repair_operators",
                ["greedy_insertion", "regret_2_insertion", "regret_k_insertion", "greedy_insertion_with_blinks"],
            ),
            operator_decay_rate=getattr(config, "operator_decay_rate", 0.95),
            score_sigma_1=getattr(config, "score_sigma_1", 33.0),
            score_sigma_2=getattr(config, "score_sigma_2", 9.0),
            score_sigma_3=getattr(config, "score_sigma_3", 3.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
