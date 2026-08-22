"""Parse slaythespire.wiki.gg ``Module:Cards/StS2_data/*`` Lua tables.

The wiki stores one Lua table per character. Each entry is::

    ["Strike (Ironclad)"] = {
        Cost = 1,
        Color = "Ironclad",
        Type = "Attack",
        Rarity = "Basic",
        Text = "Deal [6|9] damage."
    }

We do **not** keep ``Image`` (game art) or the raw ``Text`` (wiki prose).
Numbers are lifted by ``stats.extract_stats``; each wiki row becomes two
catalogue cards (base and ``+``) with distinct ids.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .ids import aliases_for, card_ids, character_slug, display_name
from .models import CatalogueCard
from .stats import extract_stats, infer_tags

_ENTRY_START = re.compile(r'\["((?:\\.|[^"\\])*)"\]\s*=\s*\{')
_STRING = re.compile(r'"((?:\\.|[^"\\])*)"')
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _skip_ws(source: str, index: int) -> int:
    n = len(source)
    while index < n and source[index] in " \t\r\n,":
        index += 1
    return index


def _parse_value(source: str, index: int) -> Tuple[Any, int]:
    index = _skip_ws(source, index)
    if index >= len(source):
        return None, index
    if source[index] == '"':
        match = _STRING.match(source, index)
        if match is None:
            return None, index + 1
        return _unescape(match.group(1)), match.end()
    if source.startswith("true", index):
        return True, index + 4
    if source.startswith("false", index):
        return False, index + 5
    if source[index] == "{":
        return _parse_table(source, index)
    num = _NUMBER.match(source, index)
    if num:
        raw = num.group(0)
        value: Any = float(raw) if "." in raw else int(raw)
        return value, num.end()
    ident = _IDENT.match(source, index)
    if ident:
        return ident.group(0), ident.end()
    return None, index + 1


def _parse_table(source: str, index: int) -> Tuple[Dict[str, Any], int]:
    """Parse a ``{ Key = value, ... }`` table. Nested tables become dicts."""
    assert source[index] == "{"
    index += 1
    table: Dict[str, Any] = {}
    n = len(source)
    while index < n:
        index = _skip_ws(source, index)
        if index < n and source[index] == "}":
            return table, index + 1
        key_match = _IDENT.match(source, index)
        if key_match is None:
            index += 1
            continue
        key = key_match.group(0)
        index = _skip_ws(source, key_match.end())
        if index < n and source[index] == "=":
            index += 1
            value, index = _parse_value(source, index)
            table[key] = value
        else:
            index += 1
    return table, index


def parse_module_entries(source: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Yield ``(wiki_name, fields)`` for every card table in a Lua module."""
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for match in _ENTRY_START.finditer(source):
        name = _unescape(match.group(1))
        fields, _end = _parse_table(source, match.end() - 1)
        entries.append((name, fields))
    return entries


def _cost_pair(fields: Dict[str, Any]) -> Tuple[float, float]:
    cost = float(fields.get("Cost") or 0)
    if "CostPlus" in fields and fields["CostPlus"] is not None:
        return cost, float(fields["CostPlus"])
    return cost, cost


def cards_from_lua(source: str, default_character: str = "unknown") -> List[CatalogueCard]:
    """Parse a wiki Lua module into base + upgraded catalogue rows."""
    cards: List[CatalogueCard] = []
    for wiki_name, fields in parse_module_entries(source):
        color = str(fields.get("Color") or default_character)
        character = character_slug(color)
        slot = str(fields.get("Type") or "Skill").lower()
        rarity = str(fields.get("Rarity") or "common").lower()
        text = str(fields.get("Text") or "")
        base_stats, up_stats = extract_stats(text)
        tags = infer_tags(text, rarity, slot)
        base_id, upgrade_id = card_ids(wiki_name, color)
        base_cost, up_cost = _cost_pair(fields)
        base_name = display_name(wiki_name, upgraded=False)
        up_name = display_name(wiki_name, upgraded=True)
        cards.append(
            CatalogueCard(
                card_id=base_id,
                name=base_name,
                character=character,
                slot=slot,
                cost=base_cost,
                rarity=rarity,
                stats=base_stats,
                tags=list(tags),
                aliases=aliases_for(wiki_name, base_name),
                upgraded=False,
                base_id=base_id,
            )
        )
        cards.append(
            CatalogueCard(
                card_id=upgrade_id,
                name=up_name,
                character=character,
                slot=slot,
                cost=up_cost,
                rarity=rarity,
                stats=up_stats,
                tags=list(tags),
                aliases=aliases_for(wiki_name, up_name),
                upgraded=True,
                base_id=base_id,
            )
        )
    return cards
