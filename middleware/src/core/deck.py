"""
Deck solution representation for deckbuilding games (e.g. Slay the Spire 2).

Unlike core.build.Build (exactly one item per slot), a Deck is a
variable-size *subset* of cards -- any number of cards may share the same
Slot/card-type. See docs/deck-problem-mapping.md for the full domain
mapping this and core.deck_problem.DeckProblem implement.

Deck subclasses Build and overrides only `equip()` and `equipped_items`.
Every other Build property (`total_cost`, `total_stats`, `tag_counts()`,
`all_tags`, `__len__`, `__iter__`, `is_valid()`) is derived purely from
`equipped_items` in the base class, so they work unmodified here -- which
is also why `core.scoring.score_build` and
`core.synergy.SynergyEngine.active_synergies` (both typed to accept a
`Build`) accept a `Deck` without any changes on their end.
"""

from typing import List

from .build import Build
from .item import Item


class Deck(Build):
    """A selected subset of cards, bounded by a target size and a gold budget.

    Attributes:
        cards: The selected cards, in insertion order.
        budget: Maximum total cost allowed (gold spent on cards).
        max_size: Maximum number of cards (the deck's target size).
    """

    def __init__(self, budget: float = float("inf"), max_size: int = 20) -> None:
        self.cards: List[Item] = []
        self.budget = budget
        self.max_size = max_size
        # Build's other derived properties don't use character_level, but
        # keep it set for isinstance(Build) callers that might read it.
        self.character_level = 99

    def equip(self, item: Item) -> bool:  # "equip" == "add this card to the deck"
        """Add a card to the deck.

        Returns:
            True if added, False if the deck is full or over budget.
        """
        if len(self.cards) >= self.max_size:
            return False
        if self.total_cost + item.cost > self.budget:
            return False
        self.cards.append(item)
        return True

    def copy(self) -> "Deck":
        new_deck = Deck(budget=self.budget, max_size=self.max_size)
        new_deck.cards = list(self.cards)
        return new_deck

    @property
    def equipped_items(self) -> List[Item]:
        """The selected cards -- Build-compatible name so score_build,
        SynergyEngine.active_synergies, and the result-JSON serializer in
        pipeline.games.optimizer all work on a Deck unmodified."""
        return self.cards

    @property
    def remaining_slots(self) -> int:
        return self.max_size - len(self.cards)

    def __repr__(self) -> str:
        names = ", ".join(item.name for item in self.cards)
        return f"Deck({names} | cost={self.total_cost:.0f}/{self.budget:.0f}, size={len(self.cards)}/{self.max_size})"
