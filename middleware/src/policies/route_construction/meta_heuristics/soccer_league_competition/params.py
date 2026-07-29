"""
Configuration parameters for the Soccer League Competition (SLC) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SLCParams:
    """
    Configuration parameters for the Soccer League Competition (SLC) solver.

    Players are individual routing solutions organised into teams.  The
    globally best solution is designated the "superstar".  Inter-team
    competition uses probabilistic match outcomes; intra-team competition
    uses local perturbations.  Stagnant teams are fully regenerated.

    Attributes:
        n_teams: Number of teams in the league.
        team_size: Number of players per team.
        max_iterations: Maximum number of seasons (outer loop).
        stagnation_limit: Seasons without improvement before team regeneration.
        n_removal: Nodes removed per intra-team perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
    """

    n_teams: int = 5
    team_size: int = 4
    max_iterations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "SLCParams":
        """Create parameters from a configuration object."""
        return cls(
            n_teams=getattr(config, "n_teams", 5),
            team_size=getattr(config, "team_size", 4),
            max_iterations=getattr(config, "max_iterations", 50),
            stagnation_limit=getattr(config, "stagnation_limit", 5),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
