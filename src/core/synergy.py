"""
Synergy system for videogame build optimization.

Synergies provide bonus stats when a build contains specific combinations
of items (identified by tags). This mirrors set bonuses in RPGs/ARPGs.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .build import Build


@dataclass
class SynergyRule:
    """A synergy rule that grants bonuses when tag thresholds are met.

    Attributes:
        name: Human-readable synergy name (e.g., "Fire Mastery").
        tag: The item tag that triggers this synergy.
        threshold: Minimum number of items with this tag to activate.
        bonus_stats: Stat bonuses granted when active.
        description: Optional flavor text.
    """

    name: str
    tag: str
    threshold: int
    bonus_stats: Dict[str, float]
    description: str = ""


@dataclass
class SynergyTier:
    """Multi-tier synergy with escalating bonuses (e.g., 2-piece / 4-piece sets).

    Attributes:
        name: Synergy set name.
        tag: The item tag for this set.
        tiers: List of (threshold, bonus_stats) pairs, ordered by threshold.
    """

    name: str
    tag: str
    tiers: List[SynergyRule] = field(default_factory=list)

    def active_tiers(self, tag_count: int) -> List[SynergyRule]:
        """Return all tiers that are currently active given the tag count."""
        return [tier for tier in self.tiers if tag_count >= tier.threshold]


class SynergyEngine:
    """Evaluates synergy bonuses for a build.

    Attributes:
        rules: List of simple synergy rules.
        tiered: List of multi-tier synergy sets.
    """

    def __init__(
        self,
        rules: List[SynergyRule] = None,
        tiered: List[SynergyTier] = None,
    ) -> None:
        self.rules: List[SynergyRule] = rules or []
        self.tiered: List[SynergyTier] = tiered or []

    def evaluate(self, build: Build) -> Dict[str, float]:
        """Compute total synergy bonus stats for a build.

        Args:
            build: The character build to evaluate.

        Returns:
            Aggregated bonus stats from all active synergies.
        """
        tag_counts = build.tag_counts()
        bonuses: Dict[str, float] = {}

        # Simple rules
        for rule in self.rules:
            count = tag_counts.get(rule.tag, 0)
            if count >= rule.threshold:
                for stat, value in rule.bonus_stats.items():
                    bonuses[stat] = bonuses.get(stat, 0.0) + value

        # Tiered synergies
        for tier_set in self.tiered:
            count = tag_counts.get(tier_set.tag, 0)
            for tier in tier_set.active_tiers(count):
                for stat, value in tier.bonus_stats.items():
                    bonuses[stat] = bonuses.get(stat, 0.0) + value

        return bonuses

    def active_synergies(self, build: Build) -> List[str]:
        """List names of all active synergies for display."""
        tag_counts = build.tag_counts()
        active: List[str] = []

        for rule in self.rules:
            if tag_counts.get(rule.tag, 0) >= rule.threshold:
                active.append(rule.name)

        for tier_set in self.tiered:
            count = tag_counts.get(tier_set.tag, 0)
            for tier in tier_set.active_tiers(count):
                active.append(tier.name)

        return active
