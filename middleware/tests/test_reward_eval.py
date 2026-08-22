"""Tests for core.reward_eval — marginal STS2 card-reward evaluation (SA3)."""

from __future__ import annotations

import ast
import inspect
from typing import List

import pytest

from core.item import Item, Rarity, Slot
from core.reward_eval import AdvisorPreferences, RunContext, evaluate_reward
from core.scoring import ScoringConfig
from core.synergy import SynergyEngine, SynergyRule


def _card(
    name: str,
    stats: dict,
    slot: Slot = Slot.ATTACK,
    rarity: Rarity = Rarity.COMMON,
    cost: float = 1.0,
    tags: set = frozenset(),
    item_id: str | None = None,
) -> Item:
    return Item(
        name=name,
        slot=slot,
        stats=stats,
        cost=cost,
        rarity=rarity,
        tags=frozenset(tags),
        item_id=item_id or name.lower().replace(" ", "_"),
    )


def _scoring() -> ScoringConfig:
    return ScoringConfig(
        stat_weights={
            "attack": 1.0,
            "block": 0.7,
            "strength_gain": 3.0,
            "multi_hit": 2.5,
            "draw": 1.0,
        },
        slot_bonus=2.0,  # must be ignored by the advisor
        rarity_bonus=1.0,
        diminishing_returns=False,
    )


def _engine() -> SynergyEngine:
    return SynergyEngine(
        rules=[
            SynergyRule(name="Strength Engine (3pc)", tag="strength", threshold=3, bonus_stats={"attack": 10.0}),
            SynergyRule(name="Multi-Hit Mastery (2pc)", tag="multi_hit", threshold=2, bonus_stats={"attack": 15.0}),
        ]
    )


def _starter() -> List[Item]:
    strikes = [_card("Strike", {"attack": 6.0}) for _ in range(5)]
    defends = [_card("Defend", {"block": 5.0}, slot=Slot.SKILL) for _ in range(4)]
    bash = [_card("Bash", {"attack": 8.0, "vulnerable": 2.0}, cost=2.0, tags={"strength"})]
    return strikes + defends + bash


class TestNoDeckProblem:
    def test_module_does_not_import_deck_problem(self) -> None:
        import core.reward_eval as module

        tree = ast.parse(inspect.getsource(module))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert all("deck_problem" not in name for name in imported)
        assert "core.deck_problem" not in imported


class TestSkipAndDuplicates:
    def test_skip_has_zero_delta_and_zero_dilution(self) -> None:
        advice = evaluate_reward(_starter(), [], scoring=_scoring(), synergies=_engine())
        assert advice.base_deck_size == 10
        skip = advice.actions[0]
        assert skip.action == "skip"
        assert skip.score_delta == pytest.approx(0.0)
        assert skip.objectives.dilution == pytest.approx(0.0)
        assert skip.objectives.synergy == pytest.approx(0.0)
        assert skip.card_name == "Skip"

    def test_duplicate_copies_are_retained(self) -> None:
        base = [_card("Strike", {"attack": 6.0}) for _ in range(5)]
        offer = _card("Strike", {"attack": 6.0})
        advice = evaluate_reward(base, [offer], scoring=_scoring())
        take = advice.actions[1]
        # Five existing Strikes plus the offer — not collapsed to one.
        assert advice.base_deck_size == 5
        assert take.action == "take"
        # Tempo density of all-Strike decks is unchanged by another Strike,
        # but dilution is strictly positive.
        assert take.objectives.dilution > 0.0


class TestSlotBonusDoesNotForceTake:
    def test_skip_is_on_the_front_when_offer_is_a_dead_draw(self) -> None:
        # Attack-dense deck; a sixth blank-stat attack should not beat Skip
        # merely because score_build used to add slot_bonus per extra card.
        base = [_card("Strike", {"attack": 6.0}) for _ in range(6)]
        junk = _card("Clash", {"attack": 0.0}, cost=0.0)
        advice = evaluate_reward(base, [junk], scoring=_scoring(), synergies=_engine())
        skip = advice.actions[0]
        take = advice.actions[1]
        assert skip.pareto_optimal
        assert take.objectives.dilution > skip.objectives.dilution
        assert "skip" in advice.pareto_front


class TestSynergyThreshold:
    def test_third_strength_card_activates_engine(self) -> None:
        base = [
            _card("Bash", {"attack": 8.0}, tags={"strength"}),
            _card("Flex", {"strength_gain": 2.0}, slot=Slot.SKILL, cost=0.0, tags={"strength"}),
            _card("Strike", {"attack": 6.0}),
        ]
        inflame = _card(
            "Inflame",
            {"strength_gain": 4.0},
            slot=Slot.POWER,
            rarity=Rarity.UNCOMMON,
            tags={"strength", "scaling"},
        )
        defend = _card("Defend", {"block": 5.0}, slot=Slot.SKILL)
        advice = evaluate_reward(base, [inflame, defend], scoring=_scoring(), synergies=_engine())
        take_inflame = next(a for a in advice.actions if a.card_id == "inflame")
        take_defend = next(a for a in advice.actions if a.card_id == "defend")
        assert "Strength Engine (3pc)" in take_inflame.synergies_gained
        assert take_inflame.objectives.synergy > take_defend.objectives.synergy
        assert take_inflame.pareto_optimal


class TestDilutionAndTypeMix:
    def test_attack_into_attack_heavy_deck_is_worse_dilution_than_a_skill(self) -> None:
        base = [_card("Strike", {"attack": 6.0}) for _ in range(8)] + [
            _card("Defend", {"block": 5.0}, slot=Slot.SKILL) for _ in range(2)
        ]
        more_attack = _card("Cleave", {"attack": 8.0, "multi_hit": 1.0}, tags={"multi_hit"})
        a_skill = _card("Shrug It Off", {"block": 8.0, "draw": 1.0}, slot=Slot.SKILL)
        advice = evaluate_reward(
            base,
            [more_attack, a_skill],
            scoring=_scoring(),
            synergies=_engine(),
            context=RunContext(act=1),
        )
        take_atk = next(a for a in advice.actions if a.card_id == "cleave")
        take_skl = next(a for a in advice.actions if a.card_id == "shrug_it_off")
        skip = advice.actions[0]
        assert take_atk.objectives.dilution > take_skl.objectives.dilution
        assert take_skl.objectives.resilience > take_atk.objectives.resilience
        assert skip.pareto_optimal or take_skl.pareto_optimal


class TestPreferencesSelectFromFront:
    def test_recommendation_is_a_pareto_point(self) -> None:
        base = _starter()
        offers = [
            _card("Inflame", {"strength_gain": 4.0}, slot=Slot.POWER, tags={"strength", "scaling"}),
            _card("Pummel", {"attack": 2.0, "multi_hit": 4.0}, rarity=Rarity.UNCOMMON, tags={"multi_hit", "strength"}),
            _card("Bludgeon", {"attack": 32.0}, rarity=Rarity.RARE, cost=3.0),
        ]
        advice = evaluate_reward(
            base,
            offers,
            scoring=_scoring(),
            synergies=_engine(),
            preferences=AdvisorPreferences(dilution_weight=2.0, tempo_weight=0.5),
        )
        assert advice.recommendation in advice.pareto_front
        assert {a.label() for a in advice.actions if a.pareto_optimal} == set(advice.pareto_front)
        assert len(advice.actions) == 4
        labels = {a.label() for a in advice.actions}
        assert "skip" in labels
        assert "bludgeon" in labels

    def test_upgrade_flag_follows_plus_suffix(self) -> None:
        advice = evaluate_reward(
            [_card("Strike", {"attack": 6.0})],
            [_card("Strike+", {"attack": 9.0})],
            scoring=_scoring(),
        )
        take = advice.actions[1]
        assert take.is_upgrade is True
        assert take.card_id == "strike+"
