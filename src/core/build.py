"""
Build solution representation for videogame build optimization.

A Build maps equipment slots to items, analogous to a vehicle route
in VRP — the solution is a selection of items subject to constraints.
"""

import copy
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set

from .item import Item, Slot


class Build:
    """A character equipment build — maps slots to equipped items.

    This is the primary solution representation, analogous to a set of
    routes in VRP. Constraints: one item per slot, total cost ≤ budget.

    Attributes:
        slots: Mapping from Slot to equipped Item (or None).
        budget: Maximum total cost allowed.
        character_level: Character level (items above this cannot be equipped).
    """

    def __init__(self, budget: float = float("inf"), character_level: int = 99) -> None:
        self.slots: Dict[Slot, Optional[Item]] = {slot: None for slot in Slot}
        self.budget = budget
        self.character_level = character_level

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def equip(self, item: Item) -> bool:
        """Equip an item into its designated slot.

        Returns:
            True if successfully equipped, False if constraints violated.
        """
        if item.level > self.character_level:
            return False

        # Check budget with this item replacing current
        current = self.slots.get(item.slot)
        current_cost = current.cost if current else 0.0
        if self.total_cost - current_cost + item.cost > self.budget:
            return False

        self.slots[item.slot] = item
        return True

    def unequip(self, slot: Slot) -> Optional[Item]:
        """Remove and return the item from a slot."""
        item = self.slots.get(slot)
        self.slots[slot] = None
        return item

    def swap(self, item: Item) -> Optional[Item]:
        """Replace the item in a slot with a new one. Returns the old item."""
        old = self.unequip(item.slot)
        if not self.equip(item):
            # Revert if new item fails constraints
            if old is not None:
                self.slots[item.slot] = old
            return None
        return old

    def copy(self) -> "Build":
        """Deep copy of this build."""
        new_build = Build(budget=self.budget, character_level=self.character_level)
        new_build.slots = copy.copy(self.slots)
        return new_build

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def equipped_items(self) -> List[Item]:
        """List of all currently equipped items."""
        return [item for item in self.slots.values() if item is not None]

    @property
    def empty_slots(self) -> List[Slot]:
        """List of slots with no item equipped."""
        return [slot for slot, item in self.slots.items() if item is None]

    @property
    def filled_slots(self) -> List[Slot]:
        """List of slots with an item equipped."""
        return [slot for slot, item in self.slots.items() if item is not None]

    @property
    def total_cost(self) -> float:
        """Total cost of all equipped items."""
        return sum(item.cost for item in self.equipped_items)

    @property
    def remaining_budget(self) -> float:
        """How much budget is left."""
        return self.budget - self.total_cost

    @property
    def total_stats(self) -> Dict[str, float]:
        """Aggregate stats across all equipped items."""
        totals: Dict[str, float] = defaultdict(float)
        for item in self.equipped_items:
            for stat, value in item.stats.items():
                totals[stat] += value
        return dict(totals)

    @property
    def all_tags(self) -> Set[str]:
        """Union of all tags from equipped items."""
        tags: Set[str] = set()
        for item in self.equipped_items:
            tags.update(item.tags)
        return tags

    def tag_counts(self) -> Dict[str, int]:
        """Count how many equipped items have each tag."""
        counts: Dict[str, int] = defaultdict(int)
        for item in self.equipped_items:
            for tag in item.tags:
                counts[tag] += 1
        return dict(counts)

    def is_valid(self) -> bool:
        """Check if build respects all constraints."""
        if self.total_cost > self.budget:
            return False
        for item in self.equipped_items:
            if item.level > self.character_level:
                return False
        return True

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.equipped_items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.equipped_items)

    def __repr__(self) -> str:
        items_str = ", ".join(f"{slot.name}={item.name}" for slot, item in self.slots.items() if item is not None)
        return f"Build({items_str} | cost={self.total_cost:.0f}/{self.budget:.0f})"
