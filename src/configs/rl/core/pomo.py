"""POMO specific configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class POMOConfig:
    """POMO specific configuration."""

    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"
