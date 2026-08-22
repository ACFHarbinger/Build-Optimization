"""Versioned JSON contract for the Slay the Spire 2 reward advisor (SA1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class CardEntry:
    """One canonical card id and its duplicate-preserving deck count."""

    card_id: str
    count: int


@dataclass(frozen=True)
class RunContextInput:
    """Optional run state; omitting it must still permit a baseline solve."""

    act: Optional[int] = None
    floor: Optional[int] = None
    hp_pct: Optional[float] = None
    gold: Optional[float] = None
    relics: List[str] = field(default_factory=list)
    potions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AdvisorPreferencesInput:
    """User-visible Pareto-front selection and seeded projection settings."""

    tempo_weight: float = 1.0
    synergy_weight: float = 1.0
    dilution_weight: float = 1.2
    resilience_weight: float = 1.0
    mc_weight: float = 0.8
    mc_rollouts: int = 200
    seed: int = 42


@dataclass(frozen=True)
class Sts2AdvisorRequest:
    """Input accepted by ``advisor_cli.py`` over stdin JSON."""

    character: str
    deck: List[CardEntry]
    offers: List[str]
    context: RunContextInput = field(default_factory=RunContextInput)
    preferences: AdvisorPreferencesInput = field(default_factory=AdvisorPreferencesInput)
    contract_version: str = CONTRACT_VERSION


def _require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _optional_number(data: Dict[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def parse_request(payload: Any) -> Sts2AdvisorRequest:
    """Validate a JSON-decoded request without accepting ambiguous state."""
    data = _require_mapping(payload, "request")
    version = data.get("contract_version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise ValueError(f"unsupported contract_version {version!r}; expected {CONTRACT_VERSION!r}")
    character = data.get("character")
    if not isinstance(character, str) or not character.strip():
        raise ValueError("character must be a non-empty string")

    raw_deck = data.get("deck")
    if not isinstance(raw_deck, list):
        raise ValueError("deck must be an array")
    deck: List[CardEntry] = []
    for index, entry in enumerate(raw_deck):
        item = _require_mapping(entry, f"deck[{index}]")
        card_id, count = item.get("card_id"), item.get("count")
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError(f"deck[{index}].card_id must be a non-empty string")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"deck[{index}].count must be a positive integer")
        deck.append(CardEntry(card_id=card_id.strip(), count=count))

    offers = data.get("offers")
    if not isinstance(offers, list) or len(offers) != 3 or any(not isinstance(x, str) or not x.strip() for x in offers):
        raise ValueError("offers must contain exactly three non-empty card identifiers")

    raw_context = _require_mapping(data.get("context", {}), "context")
    hp_pct = _optional_number(raw_context, "hp_pct")
    if hp_pct is not None and not 0.0 <= hp_pct <= 1.0:
        raise ValueError("hp_pct must be between 0 and 1")
    context = RunContextInput(
        act=int(raw_context["act"]) if raw_context.get("act") is not None else None,
        floor=int(raw_context["floor"]) if raw_context.get("floor") is not None else None,
        hp_pct=hp_pct,
        gold=_optional_number(raw_context, "gold"),
        relics=list(raw_context.get("relics") or []),
        potions=list(raw_context.get("potions") or []),
    )

    raw_preferences = _require_mapping(data.get("preferences", {}), "preferences")
    preferences = AdvisorPreferencesInput(
        tempo_weight=float(raw_preferences.get("tempo_weight", 1.0)),
        synergy_weight=float(raw_preferences.get("synergy_weight", 1.0)),
        dilution_weight=float(raw_preferences.get("dilution_weight", 1.2)),
        resilience_weight=float(raw_preferences.get("resilience_weight", 1.0)),
        mc_weight=float(raw_preferences.get("mc_weight", 0.8)),
        mc_rollouts=int(raw_preferences.get("mc_rollouts", 200)),
        seed=int(raw_preferences.get("seed", 42)),
    )
    if preferences.mc_rollouts < 1:
        raise ValueError("mc_rollouts must be positive")
    return Sts2AdvisorRequest(
        character.strip().lower(), deck, [x.strip() for x in offers], context, preferences, version
    )


def request_to_dict(request: Sts2AdvisorRequest) -> Dict[str, Any]:
    """Convert a validated request to a JSON-compatible dictionary."""
    return asdict(request)


def response_to_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the contract version to a response before JSON serialization."""
    return {"contract_version": CONTRACT_VERSION, **response}
