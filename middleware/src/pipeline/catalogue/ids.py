"""Stable catalogue ids and aliases. Display name is never the identity."""

from __future__ import annotations

import re
from typing import List, Tuple

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_NON_SLUG = re.compile(r"[^a-z0-9+]+")


def character_slug(color: str) -> str:
    """``Ironclad`` → ``ironclad``; ``The Regent`` → ``regent``."""
    text = (color or "unknown").strip().lower()
    if text.startswith("the "):
        text = text[4:]
    return _NON_SLUG.sub("_", text).strip("_") or "unknown"


def display_name(wiki_name: str, upgraded: bool = False) -> str:
    """Strip the wiki's ``(Ironclad)`` disambiguator; add ``+`` for upgrades."""
    base = _PAREN.sub("", wiki_name).strip()
    if upgraded and not base.endswith("+"):
        return base + "+"
    return base


def name_slug(wiki_name: str) -> str:
    """``Strike (Ironclad)`` → ``strike``; ``One-Two Punch`` → ``one_two_punch``."""
    base = _PAREN.sub("", wiki_name).strip().lower()
    base = base.replace("'", "").replace("'", "")
    slug = _NON_SLUG.sub("_", base).strip("_")
    return slug or "unknown"


def card_ids(wiki_name: str, color: str) -> Tuple[str, str]:
    """Return ``(base_id, upgrade_id)`` e.g. ``ironclad:strike``, ``ironclad:strike+``."""
    char = character_slug(color)
    slug = name_slug(wiki_name)
    if slug.endswith("+"):
        slug = slug[:-1].rstrip("_")
    base_id = f"{char}:{slug}"
    return base_id, f"{base_id}+"


def aliases_for(wiki_name: str, display: str) -> List[str]:
    """Approved match names: wiki title, stripped title, display name."""
    values = [wiki_name.strip(), _PAREN.sub("", wiki_name).strip(), display]
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
