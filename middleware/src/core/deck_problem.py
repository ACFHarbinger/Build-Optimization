"""
DeckProblem: Problem definition for deckbuilding-game build optimization
(e.g. Slay the Spire 2), parallel to core.problem.BuildProblem.

The critical difference from BuildProblem: equipment games assign *exactly
one* item per fixed Slot (a Multiple-Choice Knapsack). Deckbuilding games
select a variable-size *subset* of cards bounded by a target deck size --
a plain 0-1 knapsack (weight=1 per card, capacity=max_deck_size), with no
per-slot exclusivity at all. See docs/deck-problem-mapping.md for the full
mapping.

Solution representation used by solvers:
    ``selected`` -- a 1-D boolean (or 0/1 int) array of length ``num_items``,
    True/1 if that card is included in the deck.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from core.item import Item
from core.scoring import ScoringConfig, score_build
from core.synergy import SynergyEngine

if TYPE_CHECKING:
    from core.deck import Deck


class DeckProblem:
    """
    A single deck-building optimisation instance, independent of PyTorch/TorchRL.

    Attributes:
        stat_matrix:    [N, S] normalised card stats (each column in [0, 1]).
        costs:          [N] card energy/gold costs.
        rarities:       [N] rarity multiplier per card.
        stat_weights:   [S] per-stat scoring weights from ScoringConfig.
        max_deck_size:  Maximum number of cards the deck may contain.
        budget:         Maximum total cost allowed (defaults to unbounded --
                         deck *size* is normally the binding constraint, not
                         gold, matching the report's own framing).
        rarity_bonus:   Score bonus per rarity multiplier unit.
        stat_names:     Stat names aligned to stat_matrix columns.
        items:          Original Item objects (cards), needed for synergy scoring.
        scoring_config: ScoringConfig stored for reference.
        synergy_engine: Optional SynergyEngine for archetype-tag synergy bonuses.
    """

    def __init__(
        self,
        stat_matrix: np.ndarray,
        costs: np.ndarray,
        rarities: np.ndarray,
        stat_weights: np.ndarray,
        max_deck_size: int,
        budget: float = float("inf"),
        rarity_bonus: float = 0.0,
        stat_names: Optional[List[str]] = None,
        items: Optional[List[Item]] = None,
        scoring_config: Optional[ScoringConfig] = None,
        synergy_engine: Optional[SynergyEngine] = None,
    ) -> None:
        assert stat_matrix.ndim == 2, "stat_matrix must be 2-D [N, S]"
        assert costs.shape == (stat_matrix.shape[0],)
        assert rarities.shape == (stat_matrix.shape[0],)
        assert stat_weights.shape == (stat_matrix.shape[1],)

        self.stat_matrix = stat_matrix
        self.costs = costs
        self.rarities = rarities
        self.stat_weights = stat_weights
        self.max_deck_size = int(max_deck_size)
        self.budget = float(budget)
        self.rarity_bonus = float(rarity_bonus)
        self.stat_names: List[str] = stat_names or []
        self.items: List[Item] = items or []
        self.scoring_config = scoring_config
        self.synergy_engine = synergy_engine

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_items(self) -> int:
        return int(self.costs.shape[0])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_items(
        cls,
        items: List[Item],
        max_deck_size: int,
        budget: float = float("inf"),
        scoring_config: Optional[ScoringConfig] = None,
        synergy_engine: Optional[SynergyEngine] = None,
    ) -> "DeckProblem":
        """
        Construct a DeckProblem from core domain objects (cards).

        Mirrors core.problem.BuildProblem.from_items: derives the stat
        universe from the ScoringConfig plus any extra stats found on
        cards, normalises stat values to [0, 1] per column.

        Args:
            items:          Cards to optimise over (post-filter).
            max_deck_size:  Maximum number of cards the deck may contain.
            budget:         Maximum total cost allowed (default: unbounded).
            scoring_config: Scoring weights and bonuses (defaults to ScoringConfig()).
            synergy_engine: Optional synergy engine for archetype-tag bonuses.

        Returns:
            A fully populated DeckProblem ready for solvers.
        """
        cfg = scoring_config or ScoringConfig()

        stat_names: List[str] = list(cfg.stat_weights.keys())
        for it in items:
            for s in it.stats:
                if s not in stat_names:
                    stat_names.append(s)

        S = len(stat_names)
        N = len(items)

        raw = np.zeros((N, S), dtype=np.float32)
        for row, it in enumerate(items):
            for col, sname in enumerate(stat_names):
                raw[row, col] = float(it.stats.get(sname, 0.0))

        col_max = raw.max(axis=0)
        col_max[col_max == 0.0] = 1.0
        stat_matrix = raw / col_max

        stat_weights = np.array([cfg.stat_weights.get(s, 1.0) for s in stat_names], dtype=np.float32)
        costs = np.array([float(it.cost) for it in items], dtype=np.float32)
        rarities = np.array([it.rarity.multiplier for it in items], dtype=np.float32)

        return cls(
            stat_matrix=stat_matrix,
            costs=costs,
            rarities=rarities,
            stat_weights=stat_weights,
            max_deck_size=max_deck_size,
            budget=float(budget),
            rarity_bonus=float(cfg.rarity_bonus),
            stat_names=stat_names,
            items=items,
            scoring_config=cfg,
            synergy_engine=synergy_engine,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def item_value(self, item_idx: int) -> float:
        """Per-card additive score contribution (mirrors the per-item terms
        in BuildProblem.score_fast, minus the slot_bonus term -- decks have
        no slots to fill)."""
        value = float(np.dot(self.stat_matrix[item_idx], self.stat_weights))
        value += float(self.rarities[item_idx]) * self.rarity_bonus
        return value

    def score_fast(self, selected: np.ndarray) -> float:
        """
        Fast approximate score for a card selection.

        Skips synergy bonuses (which require a Deck object) but is
        suitable for use inside tight optimiser loops.

        Args:
            selected: [num_items] bool/int array; truthy means included.

        Returns:
            Scalar score (float).
        """
        idx = np.nonzero(selected)[0]
        return float(sum(self.item_value(int(i)) for i in idx))

    def score_full(self, selected: np.ndarray) -> float:
        """
        Full score via core.scoring (includes synergy bonuses).

        Converts the solver array to a Deck object and calls score_build().

        Args:
            selected: [num_items] bool/int array; truthy means included.

        Returns:
            Scalar score (float).
        """
        deck = self.to_deck(selected)
        cfg = self.scoring_config or ScoringConfig()
        return score_build(deck, self.synergy_engine, cfg)

    # ------------------------------------------------------------------
    # Solution helpers
    # ------------------------------------------------------------------

    def is_feasible(self, selected: np.ndarray) -> bool:
        """True if the selection respects both the deck-size cap and budget."""
        idx = np.nonzero(selected)[0]
        return len(idx) <= self.max_deck_size and self.budget_used(selected) <= self.budget

    def budget_used(self, selected: np.ndarray) -> float:
        """Total cost of all selected cards."""
        idx = np.nonzero(selected)[0]
        if len(idx) == 0:
            return 0.0
        return float(self.costs[idx].sum())

    def to_deck(self, selected: np.ndarray) -> "Deck":
        """Convert a solver's selection array to a core.deck.Deck."""
        from core.deck import Deck

        deck = Deck(budget=self.budget, max_size=self.max_deck_size)
        for i in np.nonzero(selected)[0]:
            deck.cards.append(self.items[int(i)])
        return deck

    def to_result_json(
        self,
        selected: np.ndarray,
        solver_name: str = "unknown",
        elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        """Convert a selection array to the standardized JSON dictionary."""
        deck = self.to_deck(selected)
        full_score = self.score_full(selected)
        active_synergies = (
            self.synergy_engine.active_synergies(deck) if self.synergy_engine else []
        )
        items = [
            {
                "id": item.item_id,
                "name": item.name,
                "slot": item.slot.name,
                "cost": item.cost,
                "rarity": item.rarity.name,
                "stats": item.stats,
                "tags": sorted(list(item.tags)),
            }
            for item in deck.equipped_items
        ]
        return {
            "solver": solver_name,
            "score": full_score,
            "cost": deck.total_cost,
            "budget": self.budget,
            "items": items,
            "synergies": active_synergies,
            "items_count": len(deck),
            "elapsed": elapsed,
            "extra": {"max_deck_size": self.max_deck_size},
        }

    # ------------------------------------------------------------------
    # Initial solutions
    # ------------------------------------------------------------------

    def greedy_solution(self) -> np.ndarray:
        """
        Deterministic greedy initial solution: takes cards in descending
        value order until the deck-size cap or budget is hit.

        Returns:
            [num_items] bool array; True means the card is selected.
        """
        selected = np.zeros(self.num_items, dtype=bool)
        order = sorted(range(self.num_items), key=lambda i: -self.item_value(i))

        count = 0
        cost_used = 0.0
        for i in order:
            if count >= self.max_deck_size:
                break
            if cost_used + float(self.costs[i]) > self.budget:
                continue
            selected[i] = True
            count += 1
            cost_used += float(self.costs[i])

        return selected
