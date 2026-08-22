"""Canonical STS2 catalogue records (SA2).

These are *our* facts table: stable ids, character tags, numeric stats.
Display name is never the identity. Wiki prose and card art are not stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.item import Item, Rarity, Slot

SLOT_MAP: Dict[str, Slot] = {
    "attack": Slot.ATTACK,
    "skill": Slot.SKILL,
    "power": Slot.POWER,
}

RARITY_MAP: Dict[str, Rarity] = {
    "basic": Rarity.COMMON,
    "common": Rarity.COMMON,
    "uncommon": Rarity.UNCOMMON,
    "rare": Rarity.RARE,
    "ancient": Rarity.LEGENDARY,
    "curse": Rarity.COMMON,
    "status": Rarity.COMMON,
    "token": Rarity.COMMON,
    "event": Rarity.COMMON,
    "quest": Rarity.COMMON,
    "starter": Rarity.COMMON,
    "epic": Rarity.EPIC,
    "legendary": Rarity.LEGENDARY,
}

WIKI_ATTRIBUTION = (
    "Card facts derived from Slay the Spire Wiki (https://slaythespire.wiki.gg), "
    "CC BY-SA. The game and its trademarks are property of Mega Crit Games / "
    "Kobold Games. This project does not redistribute wiki prose, card art, "
    "or screenshots."
)


@dataclass
class CatalogueCard:
    """One catalogue row: a base card or its upgraded ``+`` variant."""

    card_id: str
    name: str
    character: str
    slot: str
    cost: float
    rarity: str
    stats: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    upgraded: bool = False
    base_id: Optional[str] = None

    def to_item(self) -> Item:
        """Convert to a ``core.item.Item`` the evaluator already consumes."""
        rarity_key = self.rarity.lower()
        slot_key = self.slot.lower()
        tags = set(self.tags)
        tags.add(self.character)
        return Item(
            name=self.name,
            slot=SLOT_MAP.get(slot_key, Slot.SKILL),
            stats=dict(self.stats),
            cost=float(self.cost),
            rarity=RARITY_MAP.get(rarity_key, Rarity.COMMON),
            tags=frozenset(tags),
            item_id=self.card_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "character": self.character,
            "slot": self.slot,
            "cost": self.cost,
            "rarity": self.rarity,
            "stats": dict(self.stats),
            "tags": list(self.tags),
            "aliases": list(self.aliases),
            "upgraded": self.upgraded,
            "base_id": self.base_id,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CatalogueCard":
        return cls(
            card_id=str(raw["card_id"]),
            name=str(raw["name"]),
            character=str(raw.get("character") or "unknown"),
            slot=str(raw.get("slot") or "skill"),
            cost=float(raw.get("cost") or 0.0),
            rarity=str(raw.get("rarity") or "common"),
            stats={str(k): float(v) for k, v in dict(raw.get("stats") or {}).items()},
            tags=[str(t) for t in list(raw.get("tags") or [])],
            aliases=[str(a) for a in list(raw.get("aliases") or [])],
            upgraded=bool(raw.get("upgraded", False)),
            base_id=(str(raw["base_id"]) if raw.get("base_id") else None),
        )


@dataclass
class Catalogue:
    """Merged wiki-cache + local overlay."""

    cards: List[CatalogueCard]
    source: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def index(self) -> Dict[str, CatalogueCard]:
        return {card.card_id: card for card in self.cards}

    def to_items(self) -> List[Item]:
        return [card.to_item() for card in self.cards]
