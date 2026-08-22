"""Extract numeric stats from wiki.gg ``[base|upgraded]`` card text.

We persist the numbers, not the wiki prose. Heuristics are conservative:
unknown phrasing yields empty stats rather than a guess.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

_PAIR = re.compile(r"\[(\-?\d+)\|(\-?\d+)\]")
_DEAL = re.compile(r"Deal\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+damage", re.I)
_BLOCK = re.compile(r"Gain\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+\$?Block", re.I)
_STRENGTH = re.compile(r"Gain\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+\$?Strength", re.I)
_VULN = re.compile(r"Apply\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+\$?Vulnerable", re.I)
_DRAW = re.compile(r"Draw\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))", re.I)
_TIMES = re.compile(r"(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+times", re.I)
_LOSE_HP = re.compile(r"Lose\s+(?:\[(\-?\d+)\|(\-?\d+)\]|(\-?\d+))\s+HP", re.I)
_TWICE = re.compile(r"\btwice\b", re.I)


def _pair(match: re.Match[str]) -> Tuple[float, float]:
    if match.group(3) is not None:
        value = float(match.group(3))
        return value, value
    return float(match.group(1)), float(match.group(2))


def _first(pattern: re.Pattern[str], text: str) -> Tuple[float, float] | None:
    match = pattern.search(text)
    if match is None:
        return None
    return _pair(match)


def extract_stats(text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return ``(base_stats, upgraded_stats)`` parsed from a wiki Text field."""
    base: Dict[str, float] = {}
    upgraded: Dict[str, float] = {}

    def put(key: str, values: Tuple[float, float] | None) -> None:
        if values is None:
            return
        base[key] = values[0]
        upgraded[key] = values[1]

    put("attack", _first(_DEAL, text))
    put("block", _first(_BLOCK, text))
    put("strength_gain", _first(_STRENGTH, text))
    put("vulnerable", _first(_VULN, text))
    put("draw", _first(_DRAW, text))
    put("hp_loss", _first(_LOSE_HP, text))

    times = _first(_TIMES, text)
    if times is not None:
        put("multi_hit", times)
    elif _TWICE.search(text):
        base["multi_hit"] = 2.0
        upgraded["multi_hit"] = 2.0

    return base, upgraded


def infer_tags(text: str, rarity: str, slot: str) -> list[str]:
    """Cheap tag heuristics used until a hand-authored overlay overrides them."""
    tags: list[str] = []
    lowered = text.lower()
    rarity_l = rarity.lower()
    if rarity_l in {"basic", "starter"}:
        tags.append("basic")
    if "strength" in lowered:
        tags.append("strength")
    if "times" in lowered or "twice" in lowered:
        tags.append("multi_hit")
    if "at the start of your turn" in lowered and "strength" in lowered:
        tags.append("scaling")
    if slot.lower() == "power" and "strength" in lowered:
        tags.append("scaling")
    if "block" in lowered and ("body slam" in lowered or "not removed" in lowered):
        tags.append("block_synergy")
    return tags
