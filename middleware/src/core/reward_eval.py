"""
Marginal card-reward evaluation for the STS2 screenshot reward advisor (SA3).

Answers: given a fixed multiset deck ``D`` and three offered cards, how do
the four actions ``{Skip, take 1, take 2, take 3}`` compare?

This is **not** a knapsack and does **not** touch ``DeckProblem``. V1-V8's
pool-subset path (``score_fast``, ``slot_bonus`` fill-the-deck bias,
``deduplicate``) is the wrong shape for an in-run reward pick — extra cards
dilute draw, and ``Skip`` must be able to win. Evaluation reuses only
``core.scoring.score_build`` and ``SynergyEngine``, with ``slot_bonus``
forced to 0 so Take is not rewarded merely for enlarging the deck.

The four SA4 objectives (tempo, synergy delta, dilution penalty, resilience)
are computed here; ``core.pareto`` marks the non-dominated set. Projected-run
Monte-Carlo bands belong to ``core.mc_planner`` (SA5) and are layered on by
the advisor CLI (SA6), not this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.deck import Deck
from core.item import Item, Slot
from core.pareto import ADVISOR_OBJECTIVES, non_dominated_indices
from core.scoring import ScoringConfig, score_build
from core.synergy import SynergyEngine

# Stats that represent "what this deck does in the fight in front of you"
# rather than engine/scaling value. ``strength_gain`` is deliberately on the
# synergy/archetype axis, not tempo.
_TEMPO_STATS: Tuple[str, ...] = (
    "attack",
    "block",
    "draw",
    "vulnerable",
    "block_conversion",
    "multi_hit",
)


@dataclass
class RunContext:
    """Optional in-run modifiers. Missing fields must not block a baseline."""

    act: Optional[int] = None
    floor: Optional[int] = None
    hp_pct: Optional[float] = None
    gold: Optional[float] = None
    relics: Optional[List[str]] = None
    potions: Optional[List[str]] = None


@dataclass
class AdvisorPreferences:
    """Weights used only to pick a point *from* the Pareto front.

    They never collapse the front into a single opaque score: every action is
    still returned, with ``pareto_optimal`` set from the four objectives.
    """

    tempo_weight: float = 1.0
    synergy_weight: float = 1.0
    dilution_weight: float = 1.2
    resilience_weight: float = 1.0


@dataclass
class ActionObjectives:
    """The four SA4 axes for one action. Dilution is a penalty (lower better)."""

    tempo: float
    synergy: float
    dilution: float
    resilience: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "tempo": self.tempo,
            "synergy": self.synergy,
            "dilution": self.dilution,
            "resilience": self.resilience,
        }


@dataclass
class ActionAdvice:
    """Evaluation of one of {Skip, take offer i}."""

    action: str
    card_id: Optional[str]
    card_name: Optional[str]
    is_upgrade: bool
    objectives: ActionObjectives
    full_score: float
    score_delta: float
    synergies_gained: List[str] = field(default_factory=list)
    synergies_lost: List[str] = field(default_factory=list)
    tag_counts: Dict[str, int] = field(default_factory=dict)
    explanation: str = ""
    pareto_optimal: bool = False
    rank: int = 0
    weighted_score: float = 0.0

    def label(self) -> str:
        """Stable front label: ``skip`` or the taken card's id."""
        if self.action == "skip" or not self.card_id:
            return "skip"
        return self.card_id


@dataclass
class RewardAdvice:
    """Four-way (or N-offer) comparison of a confirmed reward screen."""

    base_deck_size: int
    actions: List[ActionAdvice]
    pareto_front: List[str]
    recommendation: str


def _without_slot_bonus(cfg: ScoringConfig) -> ScoringConfig:
    """Copy ``cfg`` with ``slot_bonus`` forced to 0.

    ``score_build``'s per-card slot bonus is a fill-the-deck bias that would
    make every Take beat Skip before stats matter. The advisor must not
    inherit it.
    """
    if cfg.slot_bonus == 0.0:
        return cfg
    return ScoringConfig(
        stat_weights=dict(cfg.stat_weights),
        synergy_multiplier=cfg.synergy_multiplier,
        cost_penalty=cfg.cost_penalty,
        slot_bonus=0.0,
        rarity_bonus=cfg.rarity_bonus,
        diminishing_returns=cfg.diminishing_returns,
        diminishing_threshold=cfg.diminishing_threshold,
    )


def _to_deck(cards: Sequence[Item]) -> Deck:
    """Build a Deck that preserves duplicate copies (no size/budget reject)."""
    deck = Deck(budget=float("inf"), max_size=max(len(cards) + 8, 40))
    deck.cards = list(cards)
    return deck


def _stat_value(item: Item, cfg: ScoringConfig) -> float:
    total = 0.0
    for stat, value in item.stats.items():
        total += float(value) * float(cfg.stat_weights.get(stat, 1.0))
    return total


def _tempo_score(deck: Deck, cfg: ScoringConfig) -> float:
    """Per-card immediate power (density). Skip keeps current density."""
    n = max(len(deck), 1)
    stats = deck.total_stats
    raw = 0.0
    for stat in _TEMPO_STATS:
        raw += float(stats.get(stat, 0.0)) * float(cfg.stat_weights.get(stat, 1.0))
    return raw / n


def _synergy_contribution(deck: Deck, engine: Optional[SynergyEngine], cfg: ScoringConfig) -> float:
    if engine is None:
        return 0.0
    bonuses = engine.evaluate(deck)
    score = 0.0
    for stat, value in bonuses.items():
        score += float(value) * float(cfg.stat_weights.get(stat, 1.0)) * cfg.synergy_multiplier
    return score


def _dilution_penalty(
    base: Sequence[Item],
    offer: Optional[Item],
    cfg: ScoringConfig,
    context: Optional[RunContext],
) -> float:
    """Incremental draw-slot cost of taking ``offer``. Skip is 0.0.

    Three additive terms, all zero for Skip:
    - size: every extra card costs ``1/n`` of a draw (heavier in Act 1)
    - quality gap: taking a below-average card
    - type imbalance: piling onto a type that already exceeds two-thirds
    """
    if offer is None:
        return 0.0
    n = max(len(base), 1)
    act = context.act if context is not None and context.act is not None else 1
    size_term = 1.0 / n
    if act <= 1:
        size_term *= 1.5

    avg = sum(_stat_value(c, cfg) for c in base) / n if base else 0.0
    quality_gap = max(0.0, avg - _stat_value(offer, cfg))

    same_type = sum(1 for c in base if c.slot == offer.slot)
    imbalance = 0.0
    if same_type / n >= (2.0 / 3.0):
        imbalance = 1.0 + (same_type / n - 2.0 / 3.0)

    return size_term + quality_gap + imbalance


def _resilience(deck: Deck, context: Optional[RunContext]) -> float:
    """Block/skill density, energy curve, and optional low-HP pressure."""
    n = max(len(deck), 1)
    stats = deck.total_stats
    block = float(stats.get("block", 0.0))
    attack = float(stats.get("attack", 0.0))
    block_share = block / max(block + attack, 1e-6)
    n_skills = sum(1 for c in deck.cards if c.slot == Slot.SKILL)
    skill_share = n_skills / n
    mean_cost = sum(float(c.cost) for c in deck.cards) / n
    energy_flex = 1.0 / (1.0 + max(mean_cost - 1.0, 0.0))

    hp = 100.0
    if context is not None and context.hp_pct is not None:
        hp = float(context.hp_pct)
    hp_factor = 1.0 + max(0.0, (50.0 - hp) / 100.0)

    return block_share * 2.0 * hp_factor + skill_share + energy_flex


def _is_upgrade(item: Optional[Item]) -> bool:
    if item is None:
        return False
    return item.name.endswith("+") or (item.item_id or "").endswith("+")


def _weighted(obj: ActionObjectives, prefs: AdvisorPreferences) -> float:
    return (
        prefs.tempo_weight * obj.tempo
        + prefs.synergy_weight * obj.synergy
        - prefs.dilution_weight * obj.dilution
        + prefs.resilience_weight * obj.resilience
    )


def _explain(
    action: str,
    offer: Optional[Item],
    obj: ActionObjectives,
    gained: Sequence[str],
    base_size: int,
) -> str:
    if action == "skip":
        return (
            f"Preserves current deck density ({base_size} cards). "
            "Skip is preferred when no offer earns its draw slot."
        )
    name = offer.name if offer is not None else "offer"
    bits: List[str] = [f"Take {name} into a {base_size}-card deck."]
    if gained:
        bits.append("Activates: " + ", ".join(gained) + ".")
    else:
        bits.append("No new synergy threshold crossed.")
    if obj.dilution > 0.5:
        bits.append("Draw dilution is material.")
    return " ".join(bits)


def evaluate_reward(
    base: Sequence[Item],
    offers: Sequence[Item],
    scoring: Optional[ScoringConfig] = None,
    synergies: Optional[SynergyEngine] = None,
    preferences: Optional[AdvisorPreferences] = None,
    context: Optional[RunContext] = None,
) -> RewardAdvice:
    """Score Skip and each offer as ``score_build(D)`` vs ``score_build(D+x)``.

    Args:
        base: Current deck as a multiset (duplicate copies are retained).
        offers: Candidate cards, typically three. Each is evaluated independently.
        scoring: Stat weights; ``slot_bonus`` is ignored (forced to 0).
        synergies: Optional tag-threshold engine for the synergy axis.
        preferences: Front-selection weights; does not hide dominated points.
        context: Optional Act/floor/HP/gold modifiers.

    Returns:
        ``RewardAdvice`` with every action, the non-dominated labels, and a
        recommendation chosen from the Pareto front via ``preferences``.
    """
    cfg = _without_slot_bonus(scoring or ScoringConfig())
    prefs = preferences or AdvisorPreferences()
    skip_deck = _to_deck(base)
    skip_full = score_build(skip_deck, synergies, cfg)
    skip_synergy = _synergy_contribution(skip_deck, synergies, cfg)
    skip_active = set(synergies.active_synergies(skip_deck) if synergies is not None else [])

    def evaluate_action(offer: Optional[Item]) -> ActionAdvice:
        if offer is None:
            deck = skip_deck
            action = "skip"
        else:
            deck = _to_deck(list(base) + [offer])
            action = "take"
        full = score_build(deck, synergies, cfg)
        syn = _synergy_contribution(deck, synergies, cfg)
        active = set(synergies.active_synergies(deck) if synergies is not None else [])
        gained = sorted(active - skip_active)
        lost = sorted(skip_active - active)
        obj = ActionObjectives(
            tempo=_tempo_score(deck, cfg),
            synergy=syn - skip_synergy,
            dilution=_dilution_penalty(base, offer, cfg, context),
            resilience=_resilience(deck, context),
        )
        return ActionAdvice(
            action=action,
            card_id=None if offer is None else offer.item_id,
            card_name="Skip" if offer is None else offer.name,
            is_upgrade=_is_upgrade(offer),
            objectives=obj,
            full_score=full,
            score_delta=full - skip_full,
            synergies_gained=gained,
            synergies_lost=lost,
            tag_counts=dict(deck.tag_counts()),
            explanation=_explain(action, offer, obj, gained, len(base)),
        )

    actions: List[ActionAdvice] = [evaluate_action(None)]
    for offer in offers:
        actions.append(evaluate_action(offer))

    front = set(non_dominated_indices([a.objectives.as_dict() for a in actions], ADVISOR_OBJECTIVES))
    for i, advice in enumerate(actions):
        advice.pareto_optimal = i in front
        advice.weighted_score = _weighted(advice.objectives, prefs)

    ranked = sorted(range(len(actions)), key=lambda i: -actions[i].weighted_score)
    for rank, idx in enumerate(ranked, start=1):
        actions[idx].rank = rank

    pareto_actions = [actions[i] for i in ranked if i in front]
    recommendation = pareto_actions[0].label() if pareto_actions else "skip"

    return RewardAdvice(
        base_deck_size=len(base),
        actions=actions,
        pareto_front=[actions[i].label() for i in range(len(actions)) if i in front],
        recommendation=recommendation,
    )
