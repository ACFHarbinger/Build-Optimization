"""
Data transforms for normalizing and filtering items from data sources.

Applied between data ingestion and solver input.
"""

from typing import Callable, Dict, List, Optional, Set

from src.core.item import Item, Rarity, Slot


def filter_by_level(items: List[Item], max_level: int) -> List[Item]:
    """Filter items to those equippable at or below a given level."""
    return [item for item in items if item.level <= max_level]


def filter_by_budget(items: List[Item], budget: float) -> List[Item]:
    """Filter items that individually exceed the budget."""
    return [item for item in items if item.cost <= budget]


def filter_by_rarity(
    items: List[Item],
    min_rarity: Rarity = Rarity.COMMON,
    max_rarity: Rarity = Rarity.LEGENDARY,
) -> List[Item]:
    """Filter items by rarity range."""
    return [item for item in items if min_rarity.value <= item.rarity.value <= max_rarity.value]


def filter_by_slots(items: List[Item], slots: Set[Slot]) -> List[Item]:
    """Keep only items for specific slots."""
    return [item for item in items if item.slot in slots]


def filter_by_tags(items: List[Item], required_tags: Set[str]) -> List[Item]:
    """Keep only items that have at least one of the required tags."""
    return [item for item in items if item.tags & required_tags]


def normalize_stats(
    items: List[Item],
    stat_ranges: Optional[Dict[str, float]] = None,
) -> List[Item]:
    """Normalize item stats to [0, 1] range based on max values.

    If stat_ranges is not provided, computes max from the data.
    Returns new Item instances with normalized stats.
    """
    if not items:
        return items

    # Compute max values per stat
    if stat_ranges is None:
        stat_ranges = {}
        for item in items:
            for stat, value in item.stats.items():
                stat_ranges[stat] = max(stat_ranges.get(stat, 0.0), abs(value))

    # Normalize
    normalized: List[Item] = []
    for item in items:
        new_stats = {stat: value / max(stat_ranges.get(stat, 1.0), 1e-10) for stat, value in item.stats.items()}
        normalized.append(
            Item(
                name=item.name,
                slot=item.slot,
                stats=new_stats,
                cost=item.cost,
                rarity=item.rarity,
                level=item.level,
                tags=item.tags,
                item_id=item.item_id,
            )
        )
    return normalized


def scale_by_rarity(items: List[Item]) -> List[Item]:
    """Scale item stats by their rarity multiplier.

    Returns new Item instances with scaled stats.
    """
    scaled: List[Item] = []
    for item in items:
        mult = item.rarity.multiplier
        new_stats = {stat: value * mult for stat, value in item.stats.items()}
        scaled.append(
            Item(
                name=item.name,
                slot=item.slot,
                stats=new_stats,
                cost=item.cost,
                rarity=item.rarity,
                level=item.level,
                tags=item.tags,
                item_id=item.item_id,
            )
        )
    return scaled


def deduplicate(items: List[Item]) -> List[Item]:
    """Remove duplicate items (by item_id)."""
    seen: Set[Optional[str]] = set()
    unique: List[Item] = []
    for item in items:
        if item.item_id not in seen:
            seen.add(item.item_id)
            unique.append(item)
    return unique


def compose_transforms(
    *transforms: Callable[[List[Item]], List[Item]],
) -> Callable[[List[Item]], List[Item]]:
    """Compose multiple transform functions into a single pipeline."""

    def pipeline(items: List[Item]) -> List[Item]:
        for transform in transforms:
            items = transform(items)
        return items

    return pipeline
