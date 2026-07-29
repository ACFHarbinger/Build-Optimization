"""
Item, Slot, and Rarity definitions for videogame build optimization.

These dataclasses define the domain primitives: equipment items with stats,
equipment slots, and rarity tiers.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, Optional


class Slot(Enum):
    """Equipment slots available for a character build."""

    WEAPON = auto()
    HELMET = auto()
    CHEST = auto()
    GLOVES = auto()
    BOOTS = auto()
    ACCESSORY_1 = auto()
    ACCESSORY_2 = auto()
    RING_1 = auto()
    RING_2 = auto()
    AMULET = auto()


class Rarity(Enum):
    """Item rarity tiers, each with an implicit power multiplier."""

    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5

    @property
    def multiplier(self) -> float:
        """Power multiplier associated with rarity tier."""
        return {1: 1.0, 2: 1.15, 3: 1.35, 4: 1.6, 5: 2.0}[self.value]


@dataclass(frozen=True)
class Item:
    """An equippable item in a videogame.

    Attributes:
        name: Human-readable item name.
        slot: Which equipment slot this item occupies.
        stats: Stat name → value mapping (e.g., {"attack": 50, "defense": 30}).
        cost: In-game currency cost to acquire.
        rarity: Rarity tier (affects implicit power).
        level: Minimum character level required to equip.
        tags: Categorical tags for synergy detection (e.g., {"fire", "critical"}).
        item_id: Unique identifier (auto-generated from name if not provided).
    """

    name: str
    slot: Slot
    stats: Dict[str, float]
    cost: float = 0.0
    rarity: Rarity = Rarity.COMMON
    level: int = 1
    tags: FrozenSet[str] = field(default_factory=frozenset)
    item_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.item_id is None:
            # frozen=True requires object.__setattr__
            object.__setattr__(self, "item_id", self.name.lower().replace(" ", "_"))

    def get_stat(self, stat_name: str, default: float = 0.0) -> float:
        """Get a specific stat value, returning default if not present."""
        return self.stats.get(stat_name, default)

    @property
    def total_stats(self) -> float:
        """Sum of all raw stat values."""
        return sum(self.stats.values())

    @property
    def efficiency(self) -> float:
        """Stat-per-cost efficiency ratio."""
        return self.total_stats / max(self.cost, 1.0)
