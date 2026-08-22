"""
Seeded Monte-Carlo projected-run planner for the STS2 screenshot reward
advisor.

Answers the question Grok scoped in the roadmap (bus 2026-08-22): *"what is
this pick worth over the rest of the Act?"* -- a projected-run axis that is
deliberately *not* a substitute for the four concrete
``score_build(base)`` / ``score_build(base + offer)`` evaluations in
``core.reward_eval`` (SA3) or the Pareto front in ``core.pareto`` (SA4).

The planner does **not** play combats. Exactly the roadmap's SA5
description: it samples remaining *rarity-weighted card-reward offers*
(and, if gold is provided, a minimal shop buy-vs-remove sub-model) across
the Act/floor horizon, and reports a **mean + confidence band** of the
projected final-deck score for each candidate action.

Determinism is by design and is the contract here: the same ``seed``, deck
and catalogue produce byte-for-byte identical results, so the projected
axis is reproducible across calls, across processes, and in tests. Use a
distinct ``seed`` to explore variation.

This module is intentionally *core-only*: it imports nothing from
``backend/``, ``ui/``, ``frontend/``, or ``middleware/src/pipeline``, and
no Hydra/torch. The advisor boundary (SA6) owns how the catalogue is
loaded; this module consumes plain ``core.item.Item`` objects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from core.build import Build
from core.deck import Deck
from core.item import Item, Rarity
from core.scoring import ScoringConfig, score_build
from core.synergy import SynergyEngine

# ---------------------------------------------------------------------------
# Tunables / configuration
# ---------------------------------------------------------------------------


def _default_rarity_weights() -> Dict[Rarity, float]:
    """Rarity prior for sampling future card rewards.

    Rough STS reward odds (common ≫ uncommon ≫ rare). Normalised over the
    rarities actually present in the catalogue, so a catalogue that only
    carries COMMON/UNCOMMON/RARE weights those three against one another.
    """
    return {
        Rarity.COMMON: 0.60,
        Rarity.UNCOMMON: 0.33,
        Rarity.RARE: 0.067,
        Rarity.EPIC: 0.002,
        Rarity.LEGENDARY: 0.001,
    }


@dataclass
class MCConfig:
    """Configuration for a single projected-run Monte-Carlo pass.

    Attributes:
        seed:        User-visible RNG seed. Same seed + same inputs ⇒ same bands.
        rollouts:    Number of simulated runs to aggregate.
        horizon:     Number of remaining reward/shop encounters to sample per run.
        rarity_weights: Prior weight per rarity tier for sampling future rewards.
        max_deck_size: Hard deck-size cap applied while sampling future cards.
        ci_band:     Fraction of the distribution covered by the reported band
                     (each tail gets ``(1 - ci_band) / 2``). 0.90 = the 5th-95th
                     percentile band.
        shop_prob:   Probability a future encounter is a shop instead of a free
                     card reward. Ignored when ``gold`` is not provided.
        removal_fee: Gold cost to remove a card at a shop (buy-vs-remove model).
    """

    seed: int = 42
    rollouts: int = 200
    horizon: int = 10
    rarity_weights: Dict[Rarity, float] = field(default_factory=_default_rarity_weights)
    max_deck_size: int = 18
    ci_band: float = 0.90
    shop_prob: float = 0.0
    removal_fee: float = 75.0


@dataclass
class MCResult:
    """Projected-run outcome for a single candidate action.

    Attributes:
        action:    Human-readable action label ("skip" or a card id).
        mean:      Mean projected final-deck score across ``samples`` runs.
        std:       Standard deviation of the projected score.
        ci_lower:  Lower edge of the ``ci_band`` central band.
        ci_upper:  Upper edge of the ``ci_band`` central band.
        minimum:   Worst projected score observed.
        maximum:   Best projected score observed.
        samples:   Number of runs aggregated (== ``rollouts``).
        seed:      Seed used (echoed for user visibility / reproducibility).
    """

    action: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    minimum: float
    maximum: float
    samples: int
    seed: int


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def _make_default_scorer(
    synergy_engine: Optional[SynergyEngine],
    scoring_config: Optional[ScoringConfig],
) -> Callable[[Build], float]:
    """Build a dilution-consistent deck scorer from the core scoring stack.

    The default ``core.scoring.score_build`` adds ``slot_bonus`` per
    equipped item, which Grok flagged as a systematic *fill-the-deck* bias.
    For a *projected-run* axis, an offer should not be rewarded merely for
    enlarging the deck, so ``slot_bonus`` is forced to zero unless the
    caller supplies a config that explicitly wants it. The synergy engine is
    still honoured so archetype value propagates into the projection.
    """
    cfg = scoring_config or ScoringConfig()
    if cfg.slot_bonus != 0.0:
        cfg = ScoringConfig(
            stat_weights=dict(cfg.stat_weights),
            synergy_multiplier=cfg.synergy_multiplier,
            cost_penalty=cfg.cost_penalty,
            slot_bonus=0.0,
            rarity_bonus=cfg.rarity_bonus,
            diminishing_returns=cfg.diminishing_returns,
            diminishing_threshold=cfg.diminishing_threshold,
        )

    def score(deck: Build) -> float:
        return score_build(deck, synergy_engine, cfg)

    return score


# ---------------------------------------------------------------------------
# Projected-run planner
# ---------------------------------------------------------------------------


class MonteCarloPlanner:
    """Seeded Monte-Carlo projected-run planner.

    Args:
        catalogue: The pool of cards future rewards are sampled from. Base and
            upgraded variants should both be present (they are distinct ids).
        scoring_config: Optional stat weights/bonuses; see
            ``_make_default_scorer`` for how ``slot_bonus`` is neutralised.
        synergy_engine: Optional synergy engine so archetype value feeds the
            projection.
        config: ``MCConfig`` holding seed/rollouts/horizon/weights.

    Example::

        planner = MonteCarloPlanner(
            catalogue=catalogue_items,
            synergy_engine=engine,
            config=MCConfig(seed=7, rollouts=100),
        )
        skip = planner.project(base_deck, None)          # the Skip action
        take = planner.project(base_deck, offer_card)     # add one card
        # advisor then compares skip/take means & bands with score_build deltas.
    """

    def __init__(
        self,
        catalogue: Sequence[Item],
        scoring_config: Optional[ScoringConfig] = None,
        synergy_engine: Optional[SynergyEngine] = None,
        config: Optional[MCConfig] = None,
    ) -> None:
        self.catalogue: List[Item] = list(catalogue)
        if not self.catalogue:
            raise ValueError("MonteCarloPlanner requires a non-empty catalogue")
        self.config = config or MCConfig()
        self._score = _make_default_scorer(synergy_engine, scoring_config)
        self._rarity_weights = self._normalise_rarity_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        base_deck: Sequence[Item],
        offer: Optional[Item],
        gold: Optional[float] = None,
    ) -> MCResult:
        """Project the final-deck score for taking ``offer`` (or skipping).

        Args:
            base_deck: Current deck as a multiset of cards (duplicates kept).
            offer:     Candidate card to add now, or ``None`` for the Skip
                action. The offer is added *before* sampling any future
                rewards, so all actions share the same future and differ only
                in the one-decision advantage being measured.
            gold:      If provided (and ``MCConfig.shop_prob > 0``), enables a
                minimal shop buy-vs-remove sub-model during the projection.

        Returns:
            An ``MCResult`` summary for this single candidate action.
        """
        rng = random.Random(self.config.seed)
        scores: List[float] = []

        for _ in range(self.config.rollouts):
            deck = self._start_deck(base_deck, offer)
            run_gold = gold
            for _step in range(self.config.horizon):
                deck, run_gold = self._advance(deck, run_gold, rng)
            scores.append(self._score(deck))

        return self._summarise(offer, scores)

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------

    def _normalise_rarity_weights(self) -> Callable[[random.Random], Item]:
        """Return a callable sampling a catalogue card by rarity prior.

        Weights are taken from ``config.rarity_weights``, then normalised
        over the rarities actually present in the catalogue, so a catalogue
        containing only COMMON/UNCOMMON/RARE distributes probability across
        just those tiers (and never "wastes" mass on an absent tier).
        """
        present = {it.rarity for it in self.catalogue}
        weights = {r: w for r, w in self.config.rarity_weights.items() if r in present}
        if not weights:
            # Catalogue has an unexpected rarity only; fall back to uniform.
            weights = {r: 1.0 for r in present}
        total = sum(weights.values())

        # A small closure capturing weights for the sampler below.
        def sampler(rng: random.Random) -> Item:
            pool_by_rarity: Dict[Rarity, List[Item]] = {}
            for it in self.catalogue:
                pool_by_rarity.setdefault(it.rarity, []).append(it)

            tiers = sorted(weights.keys(), key=lambda r: r.value)
            cumulative: List[float] = []
            running = 0.0
            for r in tiers:
                running += weights[r] / total
                cumulative.append(running)

            draw = rng.random()
            for tier, cum in zip(tiers, cumulative):
                if draw <= cum:
                    candidates = pool_by_rarity[tier]
                    return candidates[rng.randrange(len(candidates))]
            return self.catalogue[rng.randrange(len(self.catalogue))]

        return sampler

    def _start_deck(self, base_deck: Sequence[Item], offer: Optional[Item]) -> Deck:
        """Build the starting deck for a run: the base deck plus the offer."""
        deck = Deck(budget=float("inf"), max_size=self.config.max_deck_size)
        for card in base_deck:
            deck.cards.append(card)
        if offer is not None:
            deck.cards.append(offer)
        return deck

    def _advance(self, deck: Deck, gold: Optional[float], rng: random.Random) -> tuple[Deck, Optional[float]]:
        """Advance one projected encounter.

        With probability ``config.shop_prob`` (and only when ``gold`` is
        provided) the encounter is a shop; otherwise it is a free card reward
        the simulated "player" accepts only when it strictly improves the
        projected score (marginal, dilution-aware).

        Returns:
            The (possibly mutated) deck and the remaining gold.
        """
        is_shop = gold is not None and self.config.shop_prob > 0.0 and rng.random() < self.config.shop_prob
        if is_shop:
            return self._apply_shop(deck, gold, rng)

        card = self._sample(rng)
        if self._should_take(deck, card):
            deck.cards.append(card)
        return deck, gold

    def _apply_shop(self, deck: Deck, gold: Optional[float], rng: random.Random) -> tuple[Deck, Optional[float]]:
        """Minimal shop buy-vs-remove sub-model.

        The simulated player considers buying the sampled card (spending its
        ``cost``) and removing the single worst card (spending
        ``removal_fee``), and takes the affordable action that most improves
        the projected score. This is deliberately simple -- a *proxy* for
        "gold is future purchasing power", not a shop-price simulation.
        """
        current = self._score(deck)
        best: Optional[Deck] = None
        best_cost = 0.0

        card = self._sample(rng)
        if gold is not None and card.cost <= gold:
            bought = self._with_added(deck, card)
            if self._score(bought) > current:
                best = bought
                best_cost = card.cost

        if gold is not None and self.config.removal_fee <= gold and len(deck.cards) > 1:
            removed = self._best_removal(deck)
            if (
                removed is not None
                and self._score(removed) > current
                and (best is None or self._score(removed) > self._score(best))
            ):
                best = removed
                best_cost = self.config.removal_fee

        if best is not None:
            deck.cards = list(best.cards)
            return deck, (gold or 0.0) - best_cost
        return deck, gold

    def _should_take(self, deck: Deck, card: Item) -> bool:
        """Dilution-aware accept: only take a reward that improves the deck."""
        if len(deck.cards) >= self.config.max_deck_size:
            return False
        trial = Deck(budget=float("inf"), max_size=self.config.max_deck_size)
        trial.cards = list(deck.cards)
        trial.cards.append(card)
        return self._score(trial) > self._score(deck)

    def _best_removal(self, deck: Deck) -> Optional[Deck]:
        """Return the deck after removing the single worst card, if that helps."""
        baseline = self._score(deck)
        best: Optional[Deck] = None
        best_score = baseline
        for i in range(len(deck.cards)):
            trial = Deck(budget=float("inf"), max_size=self.config.max_deck_size)
            trial.cards = list(deck.cards[:i] + deck.cards[i + 1 :])
            cand_score = self._score(trial)
            if cand_score > best_score:
                best_score = cand_score
                best = trial
        return best

    def _with_added(self, deck: Deck, card: Item) -> Deck:
        trial = Deck(budget=float("inf"), max_size=self.config.max_deck_size)
        trial.cards = list(deck.cards)
        trial.cards.append(card)
        return trial

    def _sample(self, rng: random.Random) -> Item:
        return self._rarity_weights(rng)

    def _summarise(self, offer: Optional[Item], scores: List[float]) -> MCResult:
        ordered = sorted(scores)
        n = len(ordered)
        mean = sum(ordered) / n
        var = sum((s - mean) ** 2 for s in ordered) / n
        std = var ** 0.5

        tail = (1.0 - self.config.ci_band) / 2.0
        lo_idx = max(0, int(tail * n) - 1)
        hi_idx = min(n - 1, int((1.0 - tail) * n) - 1)
        ci_lower = ordered[lo_idx]
        ci_upper = ordered[hi_idx]

        label = offer.item_id if offer is not None else "skip"

        return MCResult(
            action=label,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            minimum=ordered[0],
            maximum=ordered[-1],
            samples=n,
            seed=self.config.seed,
        )
