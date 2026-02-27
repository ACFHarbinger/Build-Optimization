"""
Configuration parameters for the Soccer League Competition (SLC) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        time_limit: Wall-clock time limit in seconds.
    """

    n_teams: int = 5
    team_size: int = 4
    max_iterations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    time_limit: float = 60.0
