"""
VPL Configuration for Hydra.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VPLConfig:
    """
    Configuration for the Volleyball Premier League policy.

    VPL maintains a dual population structure with active and passive teams.
    The algorithm features four main phases: Team Formation, Competition,
    Substitution, and Coaching & Learning.
    """

    # VPL Population Parameters
    n_teams: int = 30  # Number of active teams (total = 2 * n_teams)
    max_iterations: int = 200
    substitution_rate: float = 0.2  # Diversity injection rate from passive teams

    # Coaching and Learning Parameters
    coaching_weight_1: float = 0.5  # Learning weight for best team
    coaching_weight_2: float = 0.3  # Learning weight for 2nd best team
    coaching_weight_3: float = 0.2  # Learning weight for 3rd best team
    elite_size: int = 3  # Number of elite teams preserved

    # Local Search Parameters
    local_search_iterations: int = 100

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Common policy fields
    vrpp: bool = True
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
