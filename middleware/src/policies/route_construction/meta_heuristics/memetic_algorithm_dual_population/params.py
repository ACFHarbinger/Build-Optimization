"""
Configuration parameters for Memetic Algorithm with Dual Population (MADP).

This is the rigorous parameter mapping for the Volleyball Premier League (VPL) algorithm
with proper Operations Research terminology.

TERMINOLOGY MAPPING (VPL → MADP):
- n_teams → population_size (number of active solutions)
- substitution_rate → diversity_injection_rate
- coaching_weight_1/2/3 → elite_learning_weights
- elite_size → elite_count

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League Algorithm."
    Applied Soft Computing, 64, 161-185. DOI: 10.1016/j.asoc.2017.11.043
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class MemeticAlgorithmDualPopulationParams:
    """
    Configuration parameters for MADP (Rigorous VPL Implementation).

    Population Structure:
        - Total solutions = 2N (N active + N reserve)
        - Active population: Competing solutions that undergo evolution
        - Reserve population: Diversity pool for diversity injection

    Core Phases:
        1. Solution Construction: Initialize 2N solutions (N active, N reserve)
        2. Competition (Ranking): Evaluate and rank active solutions by fitness
        3. Diversity Injection: Inject components from reserve population
        4. Elite-Guided Construction: Weaker solutions learn from top-k performers

    Attributes:
        population_size: Number of active solutions (N). Total = 2N with reserves.
                        (VPL: n_teams)
        max_iterations: Maximum number of evolution cycles (VPL: max_iterations).
        diversity_injection_rate: Probability of injecting diversity from reserves.
                                  (VPL: substitution_rate). Typical: 0.1-0.3.
        elite_learning_weights: Learning weights for top-k elite solutions.
                               (VPL: coaching_weight_1, coaching_weight_2, coaching_weight_3)
                               Must sum to 1.0. Best performer has highest weight.
        elite_count: Number of elite solutions preserved (VPL: elite_size).
                    Typically 3 for top-3 guidance.
        local_search_iterations: Local search refinement iterations per solution.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
    """

    # Population structure
    population_size: int = 30  # N active solutions (+ N reserve = 2N total)
    max_iterations: int = 200

    # Diversity injection (VPL: substitution)
    diversity_injection_rate: float = 0.2  # Probability of diversity injection

    # Elite-guided construction (VPL: coaching from top 3)
    elite_learning_weights: Optional[List[float]] = None  # Will be set in __post_init__
    elite_count: int = 3  # Number of elite solutions (top-k)

    # Local search refinement
    local_search_iterations: int = 100

    # Resource constraints
    time_limit: float = 300.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmDualPopulationParams":
        """Create parameters from a configuration object."""
        return cls(
            population_size=getattr(config, "population_size", 30),
            max_iterations=getattr(config, "max_iterations", 200),
            diversity_injection_rate=getattr(config, "diversity_injection_rate", 0.2),
            elite_learning_weights=getattr(config, "elite_learning_weights", None),
            elite_count=getattr(config, "elite_count", 3),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 300.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

    def __post_init__(self):
        """Validate parameter constraints and set default elite weights."""
        assert self.population_size > 0, "population_size must be positive"
        assert 0 <= self.diversity_injection_rate <= 1, "diversity_injection_rate must be in [0, 1]"
        assert self.elite_count >= 3, "elite_count must be at least 3 for elite-guided construction"

        # Set default elite learning weights if not provided
        if self.elite_learning_weights is None:
            # Default: Best gets 0.5, second-best gets 0.3, third-best gets 0.2
            self.elite_learning_weights = [0.5, 0.3, 0.2]

        # Ensure we have enough weights for elite_count
        if len(self.elite_learning_weights) < self.elite_count:
            raise ValueError(
                f"elite_learning_weights must have at least {self.elite_count} values, "
                f"got {len(self.elite_learning_weights)}"
            )

        # Validate elite learning weights sum to 1.0
        total_weight = sum(self.elite_learning_weights[: self.elite_count])
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Elite learning weights (first {self.elite_count}) must sum to 1.0, got {total_weight}")

    # ------------------------------------------------------------------
    # Compatibility aliases for EXACT matching with VPL attribute names
    # ------------------------------------------------------------------

    @property
    def substitution_rate(self) -> float:
        """Alias for diversity_injection_rate to match VPL exactly."""
        return self.diversity_injection_rate

    @property
    def elite_size(self) -> int:
        """Alias for elite_count to match VPL exactly."""
        return self.elite_count

    @property
    def coaching_weight_1(self) -> float:
        """Alias for first elite learning weight to match VPL exactly."""
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 1
        return self.elite_learning_weights[0]

    @property
    def coaching_weight_2(self) -> float:
        """Alias for second elite learning weight to match VPL exactly."""
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 2
        return self.elite_learning_weights[1]

    @property
    def coaching_weight_3(self) -> float:
        """Alias for third elite learning weight to match VPL exactly."""
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 3
        return self.elite_learning_weights[2]
