from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ABPCHGConfig:
    """Config for Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG)."""

    gamma: float = 0.95
    seed: Optional[int] = None
