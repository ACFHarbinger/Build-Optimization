"""
Game profile configuration.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.scoring import ScoringConfig


@dataclass
class GameDataConfig:
    """Game data paths."""

    items_path: str = "src/data/sample_games/rpg.json"
    synergies_path: Optional[str] = None


@dataclass
class GameConfig:
    """Game profile configuration.

    Attributes:
        name: Name of the game profile.
        description: Description of the game profile.
        data: Data path configuration.
        scoring: Scoring configuration and stat weights.
    """

    name: str = "Fantasy RPG"
    description: str = "Classic RPG with equipment slots, stat synergies, and budget constraints"
    data: GameDataConfig = field(default_factory=GameDataConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
