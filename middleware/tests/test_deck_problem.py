"""
Tests for core.deck_problem.DeckProblem and pipeline.decks.optimizer.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np
import pytest

from core.deck import Deck
from core.deck_problem import DeckProblem
from core.item import Item, Rarity, Slot
from core.scoring import ScoringConfig
from core.synergy import SynergyEngine, SynergyRule
from pipeline.decks import run_deck_optimization


@pytest.fixture
def sample_cards() -> List[Item]:
    return [
        Item(name="Strike", slot=Slot.ATTACK, stats={"attack": 6.0}, cost=1.0, rarity=Rarity.COMMON, item_id="c1"),
        Item(name="Defend", slot=Slot.SKILL, stats={"block": 5.0}, cost=1.0, rarity=Rarity.COMMON, item_id="c2"),
        Item(name="Demon Form", slot=Slot.POWER, stats={"strength_gain": 6.0}, cost=3.0, rarity=Rarity.RARE, tags=frozenset({"strength", "scaling"}), item_id="c3"),
        Item(name="Pummel", slot=Slot.ATTACK, stats={"attack": 2.0, "multi_hit": 4.0}, cost=1.0, rarity=Rarity.UNCOMMON, tags=frozenset({"multi_hit", "strength"}), item_id="c4"),
        Item(name="Inflame", slot=Slot.POWER, stats={"strength_gain": 4.0}, cost=1.0, rarity=Rarity.UNCOMMON, tags=frozenset({"strength", "scaling"}), item_id="c5"),
    ]


class TestDeckProblem:
    def test_from_items_and_dimensions(self, sample_cards: List[Item]) -> None:
        cfg = ScoringConfig(stat_weights={"attack": 1.0, "strength_gain": 3.0, "multi_hit": 2.0})
        problem = DeckProblem.from_items(items=sample_cards, max_deck_size=3, scoring_config=cfg)

        assert problem.num_items == 5
        assert problem.max_deck_size == 3
        # 4 distinct stats present across sample cards: attack, block, strength_gain, multi_hit
        assert problem.stat_matrix.shape == (5, 4)

    def test_item_value_and_greedy_solution(self, sample_cards: List[Item]) -> None:
        cfg = ScoringConfig(stat_weights={"attack": 1.0, "strength_gain": 3.0, "multi_hit": 2.0}, rarity_bonus=1.0)
        problem = DeckProblem.from_items(items=sample_cards, max_deck_size=2, scoring_config=cfg)

        selected = problem.greedy_solution()
        assert selected.sum() == 2
        assert problem.is_feasible(selected)

    def test_brute_force_cross_check(self, sample_cards: List[Item]) -> None:
        """Brute-force all combinations of size <= max_deck_size and check that
        our greedy/knapsack solver matches or approaches brute-force optimum."""
        cfg = ScoringConfig(stat_weights={"attack": 1.0, "strength_gain": 3.0, "multi_hit": 2.0}, rarity_bonus=1.0)
        max_size = 3
        problem = DeckProblem.from_items(items=sample_cards, max_deck_size=max_size, scoring_config=cfg)

        # Brute force search
        best_score = -1.0
        best_combo = None

        for r in range(1, max_size + 1):
            for combo in itertools.combinations(range(len(sample_cards)), r):
                vec = np.zeros(len(sample_cards), dtype=bool)
                vec[list(combo)] = True
                score = problem.score_fast(vec)
                if score > best_score:
                    best_score = score
                    best_combo = vec

        solver_selected = problem.greedy_solution()
        solver_score = problem.score_fast(solver_selected)

        assert solver_score == pytest.approx(best_score)

    def test_to_deck_and_result_json(self, sample_cards: List[Item]) -> None:
        synergies = [
            SynergyRule(name="Strength Synergies", tag="strength", threshold=2, bonus_stats={"attack": 10.0})
        ]
        engine = SynergyEngine(rules=synergies)
        cfg = ScoringConfig(stat_weights={"attack": 1.0, "strength_gain": 3.0})
        problem = DeckProblem.from_items(items=sample_cards, max_deck_size=3, scoring_config=cfg, synergy_engine=engine)

        selected = np.array([True, False, True, True, False])
        deck = problem.to_deck(selected)

        assert isinstance(deck, Deck)
        assert len(deck.equipped_items) == 3
        assert "c1" in [c.item_id for c in deck.equipped_items]

        result_json = problem.to_result_json(selected, solver_name="test_solver")
        assert result_json["solver"] == "test_solver"
        assert result_json["items_count"] == 3
        assert len(result_json["items"]) == 3
        assert "Strength Synergies" in result_json["synergies"]


class TestDeckPipeline:
    def test_run_deck_optimization_integration(self) -> None:
        items_path = "src/data/sample/slay_the_spire_2_ironclad.json"
        result = run_deck_optimization(
            solver_name="greedy",
            items_path=items_path,
            max_deck_size=16,
            verbose=False,
            persist=False,
        )

        assert result["success"] is True
        assert result["solver"] == "greedy"
        assert result["items_equipped"] == 16
        assert len(result["build"].equipped_items) == 16
