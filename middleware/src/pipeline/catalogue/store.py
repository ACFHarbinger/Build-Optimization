"""Gitignored wiki cache + local overlay (SA2).

Default location is the user app-data dir (or ``$STS2_CATALOGUE_DIR``).
Nothing written here is committed. Overlay rows win on ``card_id``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import WIKI_ATTRIBUTION, Catalogue, CatalogueCard

CACHE_FILENAME = "wiki_cache.json"
OVERLAY_FILENAME = "overlay.json"


def default_data_dir() -> Path:
    """User-local catalogue dir. Override with ``STS2_CATALOGUE_DIR``."""
    override = os.environ.get("STS2_CATALOGUE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "build-optimization" / "sts2"
    return Path.home() / ".local" / "share" / "build-optimization" / "sts2"


class CatalogueStore:
    """Load/save the wiki cache and merge a local overlay on top."""

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.cache_path = self.data_dir / CACHE_FILENAME
        self.overlay_path = self.data_dir / OVERLAY_FILENAME

    def load_cache(self) -> List[CatalogueCard]:
        if not self.cache_path.is_file():
            return []
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        return [CatalogueCard.from_dict(row) for row in payload.get("cards", [])]

    def load_overlay(self) -> List[CatalogueCard]:
        if not self.overlay_path.is_file():
            return []
        payload = json.loads(self.overlay_path.read_text(encoding="utf-8"))
        return [CatalogueCard.from_dict(row) for row in payload.get("cards", [])]

    def load(self) -> Catalogue:
        """Wiki cache, then overlay (add *and* override by ``card_id``)."""
        merged = _overlay(self.load_cache(), self.load_overlay())
        source = {}
        if self.cache_path.is_file():
            source = json.loads(self.cache_path.read_text(encoding="utf-8")).get("source", {})
        return Catalogue(cards=merged, source=source, schema_version=1)

    def write_cache(self, cards: Iterable[CatalogueCard], source: Optional[Dict[str, Any]] = None) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "source": source
            or {
                "wiki": "https://slaythespire.wiki.gg",
                "attribution": WIKI_ATTRIBUTION,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            "cards": [card.to_dict() for card in cards],
        }
        self.cache_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return self.cache_path

    def upsert_overlay(self, card: CatalogueCard) -> Path:
        """Add or replace one local row. Overlay always wins on ``card_id``."""
        existing = {row.card_id: row for row in self.load_overlay()}
        if card.card_id in existing:
            existing[card.card_id] = _merge_card(existing[card.card_id], card)
        else:
            existing[card.card_id] = card
        return self._write_overlay(list(existing.values()))

    def _write_overlay(self, cards: List[CatalogueCard]) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "cards": [card.to_dict() for card in cards]}
        self.overlay_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.overlay_path


def _merge_card(base: CatalogueCard, overlay: CatalogueCard) -> CatalogueCard:
    """Overlay fields replace wiki fields; empty overlay lists do not wipe."""
    data = base.to_dict()
    incoming = overlay.to_dict()
    for key, value in incoming.items():
        if key in {"stats"} and value:
            merged_stats = dict(data.get("stats") or {})
            merged_stats.update(value)
            data["stats"] = merged_stats
        elif key in {"tags", "aliases"} and value:
            data[key] = list(dict.fromkeys(list(data.get(key) or []) + list(value)))
        elif key in {"name", "character", "slot", "rarity", "cost", "upgraded", "base_id", "card_id"}:
            if value not in (None, "", [], {}):
                data[key] = value
    return CatalogueCard.from_dict(data)


def _overlay(cache: List[CatalogueCard], overlay: List[CatalogueCard]) -> List[CatalogueCard]:
    by_id = {card.card_id: card for card in cache}
    for card in overlay:
        if card.card_id in by_id:
            by_id[card.card_id] = _merge_card(by_id[card.card_id], card)
        else:
            by_id[card.card_id] = card
    return list(by_id.values())
