"""
Tests for the core domain model.
"""

import pytest

from src.core.build import Build
from src.core.item import Item, Rarity, Slot
from src.core.scoring import ScoringConfig, score_build
from src.core.synergy import SynergyEngine, SynergyRule


# =============================================================================
# Item Tests
# =============================================================================


class TestItem:
    def test_create_item(self) -> None:
        item = Item(
            name="Iron Sword",
            slot=Slot.WEAPON,
            stats={"attack": 50, "speed": 10},
            cost=100,
            rarity=Rarity.COMMON,
        )
        assert item.name == "Iron Sword"
        assert item.slot == Slot.WEAPON
        assert item.get_stat("attack") == 50
        assert item.get_stat("defense") == 0.0
        assert item.total_stats == 60
        assert item.item_id == "iron_sword"

    def test_item_efficiency(self) -> None:
        item = Item(name="Test", slot=Slot.WEAPON, stats={"attack": 100}, cost=50)
        assert item.efficiency == 2.0

    def test_rarity_multiplier(self) -> None:
        assert Rarity.COMMON.multiplier == 1.0
        assert Rarity.LEGENDARY.multiplier == 2.0

    def test_item_frozen(self) -> None:
        item = Item(name="Test", slot=Slot.WEAPON, stats={"attack": 10})
        with pytest.raises(AttributeError):
            item.name = "Modified"  # type: ignore[misc]


# =============================================================================
# Build Tests
# =============================================================================


class TestBuild:
    def test_equip_item(self) -> None:
        build = Build(budget=1000)
        item = Item(name="Sword", slot=Slot.WEAPON, stats={"attack": 50}, cost=100)
        assert build.equip(item) is True
        assert build.slots[Slot.WEAPON] == item
        assert build.total_cost == 100

    def test_budget_constraint(self) -> None:
        build = Build(budget=50)
        item = Item(name="Expensive", slot=Slot.WEAPON, stats={"attack": 100}, cost=100)
        assert build.equip(item) is False

    def test_level_constraint(self) -> None:
        build = Build(budget=1000, character_level=5)
        item = Item(name="High Level", slot=Slot.WEAPON, stats={"attack": 100}, level=10)
        assert build.equip(item) is False

    def test_unequip(self) -> None:
        build = Build(budget=1000)
        item = Item(name="Sword", slot=Slot.WEAPON, stats={"attack": 50}, cost=100)
        build.equip(item)
        removed = build.unequip(Slot.WEAPON)
        assert removed == item
        assert build.slots[Slot.WEAPON] is None
        assert build.total_cost == 0

    def test_swap(self) -> None:
        build = Build(budget=1000)
        item1 = Item(name="Sword1", slot=Slot.WEAPON, stats={"attack": 50}, cost=100)
        item2 = Item(name="Sword2", slot=Slot.WEAPON, stats={"attack": 80}, cost=200)
        build.equip(item1)
        old = build.swap(item2)
        assert old == item1
        assert build.slots[Slot.WEAPON] == item2

    def test_total_stats(self) -> None:
        build = Build(budget=1000)
        build.equip(Item(name="S", slot=Slot.WEAPON, stats={"attack": 50}))
        build.equip(Item(name="H", slot=Slot.HELMET, stats={"attack": 10, "defense": 30}))
        stats = build.total_stats
        assert stats["attack"] == 60
        assert stats["defense"] == 30

    def test_copy(self) -> None:
        build = Build(budget=1000)
        build.equip(Item(name="S", slot=Slot.WEAPON, stats={"attack": 50}))
        copy = build.copy()
        copy.unequip(Slot.WEAPON)
        assert build.slots[Slot.WEAPON] is not None  # Original unchanged

    def test_tag_counts(self) -> None:
        build = Build(budget=10000)
        build.equip(Item(name="A", slot=Slot.WEAPON, stats={}, tags=frozenset({"fire", "melee"})))
        build.equip(Item(name="B", slot=Slot.HELMET, stats={}, tags=frozenset({"fire"})))
        counts = build.tag_counts()
        assert counts["fire"] == 2
        assert counts["melee"] == 1


# =============================================================================
# Synergy Tests
# =============================================================================


class TestSynergy:
    def test_synergy_activation(self) -> None:
        rule = SynergyRule(
            name="Fire Mastery",
            tag="fire",
            threshold=2,
            bonus_stats={"attack": 20},
        )
        engine = SynergyEngine(rules=[rule])

        build = Build(budget=10000)
        build.equip(Item(name="A", slot=Slot.WEAPON, stats={"attack": 10}, tags=frozenset({"fire"})))
        build.equip(Item(name="B", slot=Slot.HELMET, stats={"attack": 10}, tags=frozenset({"fire"})))

        bonuses = engine.evaluate(build)
        assert bonuses["attack"] == 20

    def test_synergy_inactive(self) -> None:
        rule = SynergyRule(name="Fire", tag="fire", threshold=3, bonus_stats={"attack": 20})
        engine = SynergyEngine(rules=[rule])

        build = Build(budget=10000)
        build.equip(Item(name="A", slot=Slot.WEAPON, stats={}, tags=frozenset({"fire"})))

        bonuses = engine.evaluate(build)
        assert "attack" not in bonuses


# =============================================================================
# Scoring Tests
# =============================================================================


class TestScoring:
    def test_score_empty_build(self) -> None:
        build = Build(budget=1000)
        assert score_build(build) == 0.0

    def test_score_with_items(self) -> None:
        build = Build(budget=1000)
        build.equip(Item(name="S", slot=Slot.WEAPON, stats={"attack": 50}, rarity=Rarity.RARE))
        score = score_build(build)
        assert score > 0

    def test_synergy_increases_score(self) -> None:
        rule = SynergyRule(name="Fire", tag="fire", threshold=2, bonus_stats={"attack": 50})
        engine = SynergyEngine(rules=[rule])

        build = Build(budget=10000)
        build.equip(Item(name="A", slot=Slot.WEAPON, stats={"attack": 10}, tags=frozenset({"fire"})))
        build.equip(Item(name="B", slot=Slot.HELMET, stats={"attack": 10}, tags=frozenset({"fire"})))

        score_without = score_build(build, synergy_engine=None)
        score_with = score_build(build, synergy_engine=engine)
        assert score_with > score_without
