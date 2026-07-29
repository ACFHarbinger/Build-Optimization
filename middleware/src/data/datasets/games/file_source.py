"""
File-based data source for loading items from JSON/CSV files.

This is the primary data source for offline datasets and sample games.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.item import Item, Rarity, Slot
from core.synergy import SynergyRule

from .base import DataSource


def _resolve_data_path(raw_path: str) -> Path:
    """Resolve a config-relative data path (e.g. ``src/data/sample/rpg.json``).

    Game configs write paths relative to ``middleware/`` (their own source
    tree). Try, in order: as given (absolute, or relative to the current
    working directory), then relative to ``middleware/`` under the repo root.
    """
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path

    try:
        from constants import ROOT_DIR

        candidate = Path(ROOT_DIR) / "middleware" / raw_path
        if candidate.exists():
            return candidate
    except ImportError:
        pass

    return path


class FileSource(DataSource):
    """Load items and synergies from local JSON or CSV files.

    Supports:
        - JSON: structured item/synergy data
        - CSV: tabular item data

    Attributes:
        items_path: Path to the items data file.
        synergies_path: Optional path to synergies data file.
    """

    # Mapping from string names to Slot enum
    SLOT_MAP: Dict[str, Slot] = {
        "weapon": Slot.WEAPON,
        "helmet": Slot.HELMET,
        "chest": Slot.CHEST,
        "gloves": Slot.GLOVES,
        "boots": Slot.BOOTS,
        "accessory_1": Slot.ACCESSORY_1,
        "accessory_2": Slot.ACCESSORY_2,
        "ring_1": Slot.RING_1,
        "ring_2": Slot.RING_2,
        "amulet": Slot.AMULET,
    }

    RARITY_MAP: Dict[str, Rarity] = {
        "common": Rarity.COMMON,
        "uncommon": Rarity.UNCOMMON,
        "rare": Rarity.RARE,
        "epic": Rarity.EPIC,
        "legendary": Rarity.LEGENDARY,
    }

    def __init__(
        self,
        items_path: str,
        synergies_path: Optional[str] = None,
    ) -> None:
        self.items_path = _resolve_data_path(items_path)
        self.synergies_path = _resolve_data_path(synergies_path) if synergies_path else None

    def fetch_items(self) -> List[Item]:
        """Load items from file.

        JSON format expected:
        {
            "items": [
                {
                    "name": "Iron Sword",
                    "slot": "weapon",
                    "stats": {"attack": 50, "speed": 10},
                    "cost": 100,
                    "rarity": "common",
                    "level": 1,
                    "tags": ["melee", "iron"]
                },
                ...
            ]
        }
        """
        if self.items_path.suffix == ".json":
            return self._load_items_json()
        elif self.items_path.suffix == ".csv":
            return self._load_items_csv()
        else:
            raise ValueError(f"Unsupported file format: {self.items_path.suffix}")

    def fetch_synergies(self) -> List[SynergyRule]:
        """Load synergy rules from file.

        JSON format expected:
        {
            "synergies": [
                {
                    "name": "Fire Mastery",
                    "tag": "fire",
                    "threshold": 3,
                    "bonus_stats": {"attack": 20, "critical_rate": 5}
                },
                ...
            ]
        }
        """
        if self.synergies_path is None:
            # Try to find synergies in the items file
            if self.items_path.suffix == ".json":
                return self._load_synergies_from_items_json()
            return []

        if not self.synergies_path.exists():
            return []

        with open(self.synergies_path, "r") as f:
            data = json.load(f)

        return [
            SynergyRule(
                name=s["name"],
                tag=s["tag"],
                threshold=s["threshold"],
                bonus_stats=s["bonus_stats"],
                description=s.get("description", ""),
            )
            for s in data.get("synergies", [])
        ]

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_items_json(self) -> List[Item]:
        with open(self.items_path, "r") as f:
            data = json.load(f)

        items: List[Item] = []
        for entry in data.get("items", []):
            items.append(self._parse_item(entry))
        return items

    def _load_items_csv(self) -> List[Item]:
        items: List[Item] = []
        with open(self.items_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse stats from columns that aren't metadata
                meta_cols = {"name", "slot", "cost", "rarity", "level", "tags"}
                stats = {k: float(v) for k, v in row.items() if k not in meta_cols and v}
                entry = {
                    "name": row["name"],
                    "slot": row["slot"],
                    "stats": stats,
                    "cost": float(row.get("cost", 0)),
                    "rarity": row.get("rarity", "common"),
                    "level": int(row.get("level", 1)),
                    "tags": row.get("tags", "").split(",") if row.get("tags") else [],
                }
                items.append(self._parse_item(entry))
        return items

    def _load_synergies_from_items_json(self) -> List[SynergyRule]:
        with open(self.items_path, "r") as f:
            data = json.load(f)
        return [
            SynergyRule(
                name=s["name"],
                tag=s["tag"],
                threshold=s["threshold"],
                bonus_stats=s["bonus_stats"],
                description=s.get("description", ""),
            )
            for s in data.get("synergies", [])
        ]

    def _parse_item(self, entry: Dict[str, Any]) -> Item:
        slot_str = entry["slot"].lower()
        rarity_str = entry.get("rarity", "common").lower()
        tags = entry.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        return Item(
            name=entry["name"],
            slot=self.SLOT_MAP.get(slot_str, Slot.WEAPON),
            stats=entry.get("stats", {}),
            cost=float(entry.get("cost", 0)),
            rarity=self.RARITY_MAP.get(rarity_str, Rarity.COMMON),
            level=int(entry.get("level", 1)),
            tags=frozenset(tags),
        )
