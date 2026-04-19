"""
Configuration parameters for the Volleyball Premier League (VPL) algorithm.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VPLParams:
    """
    Configuration parameters for the Volleyball Premier League algorithm.

    VPL is a population-based metaheuristic inspired by professional volleyball
    season dynamics. The algorithm maintains a hierarchical population structure
    with active teams (competing) and passive teams (reserves for diversity).

    Population Structure:
        - Total teams = 2N (N active + N passive)
        - Active teams: Compete, evolve, and undergo coaching
        - Passive teams: Reserve pool for diversity injection via substitution

    Core Phases:
        1. Team Formation: Initialize 2N teams (N active, N passive)
        2. Competition (Racing): Evaluate and rank active teams
        3. Substitution: Inject diversity from passive teams
        4. Coaching and Learning: Weaker teams learn from top 3 performers

    Attributes:
        n_teams: Number of active teams (N). Total population = 2N.
        max_iterations: Maximum number of VPL seasons (iterations).
        substitution_rate: Probability of substituting a solution component
                          from passive teams [0, 1]. Typical: 0.1-0.3.
        coaching_weight_1: Learning weight for best team (typically 0.5).
        coaching_weight_2: Learning weight for 2nd best team (typically 0.3).
        coaching_weight_3: Learning weight for 3rd best team (typically 0.2).
        elite_size: Number of elite teams preserved (typically 3).
        local_search_iterations: Number of local search iterations per team.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
    """

    n_teams: int = 30
    max_iterations: int = 200
    substitution_rate: float = 0.2
    coaching_weight_1: float = 0.5
    coaching_weight_2: float = 0.3
    coaching_weight_3: float = 0.2
    elite_size: int = 3
    local_search_iterations: int = 100
    time_limit: float = 300.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    def __post_init__(self):
        """Validate parameter constraints."""
        assert self.n_teams > 0, "n_teams must be positive"
        assert 0 <= self.substitution_rate <= 1, "substitution_rate must be in [0, 1]"
        assert self.elite_size >= 1, "elite_size must be at least 1"

        # Validate coaching weights sum to 1.0
        total_weight = self.coaching_weight_1 + self.coaching_weight_2 + self.coaching_weight_3
        assert abs(total_weight - 1.0) < 1e-6, f"Coaching weights must sum to 1.0, got {total_weight}"

    @classmethod
    def from_config(cls, config: Any) -> "VPLParams":
        """Create parameters from a configuration object."""
        return cls(
            n_teams=getattr(config, "n_teams", 30),
            max_iterations=getattr(config, "max_iterations", 200),
            substitution_rate=getattr(config, "substitution_rate", 0.2),
            coaching_weight_1=getattr(config, "coaching_weight_1", 0.5),
            coaching_weight_2=getattr(config, "coaching_weight_2", 0.3),
            coaching_weight_3=getattr(config, "coaching_weight_3", 0.2),
            elite_size=getattr(config, "elite_size", 3),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 300.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
