"""
Tests for core.recognition — the RecognizedName seam / SA10.

Pure, no OCR dependency: exercises normalization, exact→alias→fuzzy matching
(including the space-insensitive fuzzy path), the confidence policy, and the
unknown-card blocking branch. Deterministic.
"""

from __future__ import annotations

from typing import List, Tuple

from core.recognition import (
    ConfidencePolicy,
    MatchMethod,
    NameMatcher,
    RecognizedName,
    apply_confidence,
    normalize_name,
    resolve_name,
    similarity_score,
)


def _catalogue() -> List[Tuple[str, str]]:
    return [
        ("carnage", "Carnage"),
        ("inflame", "Inflame"),
        ("cleave", "Cleave"),
        ("pommel_strike", "Pommel Strike"),
        ("demon_form", "Demon Form"),
        ("bludgeon", "Bludgeon"),
        ("spot_weakness", "Spot Weakness"),
    ]


class TestNormalize:
    def test_preserves_upgrade_plus(self) -> None:
        assert normalize_name("Carnage+") == "carnage+"
        assert normalize_name("Carnage") == "carnage"
        assert normalize_name("carnage+") != normalize_name("carnage")

    def test_strips_punctuation_and_folds_unicode(self) -> None:
        assert normalize_name("  Pommel Strike, ") == "pommel strike"
        assert normalize_name("Ｓｔｒｉｋｅ") == "strike"


class TestSimilarity:
    def test_identical_and_disjoint(self) -> None:
        assert similarity_score("carnage", "carnage") == 1.0
        assert similarity_score("a", "b") == 0.0

    def test_single_insert_space_setup(self) -> None:
        # The case the space-insensitive matcher handles: an OCR-injected space.
        assert similarity_score("inflame", "inflame") == 1.0


class TestNameMatcher:
    def test_exact_exact_alias_fuzzy(self) -> None:
        matcher = NameMatcher(
            _catalogue(),
            aliases={"inflame": ["Firebrand"]},
        )
        # exact
        card_id, method, _ = matcher.resolve("Carnage")
        assert (card_id, method) == ("carnage", MatchMethod.EXACT)
        # alias
        card_id, method, _ = matcher.resolve("firebrand")
        assert (card_id, method) == ("inflame", MatchMethod.ALIAS)
        # fuzzy (space-insensitive — OCR split the name)
        card_id, method, cands = matcher.resolve("Inf lame")
        assert (card_id, method) == ("inflame", MatchMethod.FUZZY)
        assert cands  # still returns candidates for the UI

    def test_upgrade_is_distinct(self) -> None:
        matcher = NameMatcher(_catalogue())
        # "Carnage+" is not in catalogue → unresolved (needs_dataset_entry).
        card_id, _, cands = matcher.resolve("carnage+")
        assert card_id is None
        assert cands  # the bounded candidate list is offered for a manual pick

    def test_near_tie_not_silently_picked(self) -> None:
        # "spot weakness" with an OCR-injected space still resolves cleanly
        # (space-insensitive exact match on the compact form).
        matcher = NameMatcher([("spot_weakness", "Spot Weakness"), ("cleave", "Cleave")])
        card_id, method, _ = matcher.resolve("spot weakness")
        assert card_id == "spot_weakness"
        assert method == MatchMethod.EXACT

    def test_ambiguous_match_returns_none_not_a_silent_pick(self) -> None:
        # Two genuinely-close candidates tie → NO silent auto-pick: None + the
        # bounded candidate list for a manual decision.
        matcher = NameMatcher(
            [
                ("spot_weakness", "Spot Weakness"),
                ("spot_weakness_alt", "Spot WeaKness"),
            ]
        )
        card_id, method, cands = matcher.resolve("spot wealness")
        assert card_id is None  # near-tie → unresolved, must be surfaced
        assert method == MatchMethod.FUZZY
        assert len(cands) <= 3

    def test_unknown_returns_none_and_candidates(self) -> None:
        matcher = NameMatcher(_catalogue())
        card_id, method, cands = matcher.resolve("Qliphoth")
        assert card_id is None
        assert method == MatchMethod.FUZZY
        assert len(cands) <= 3  # bounded


class TestConfidencePolicy:
    def test_confident_fuzzy_is_tentative(self) -> None:
        name = RecognizedName(
            region_id="offer1", raw_text="Inf lame", normalized="inf lame",
            confidence=0.98, method=MatchMethod.FUZZY, matched_card_id="inflame",
        )
        assert apply_confidence(name, accept_threshold=0.90) == ConfidencePolicy.TENTATIVE

    def test_low_confident_fuzzy_blocks(self) -> None:
        name = RecognizedName(
            region_id="offer1", raw_text="Whoknows", normalized="whoknows",
            confidence=0.70, method=MatchMethod.FUZZY, matched_card_id="bludgeon",
        )
        assert apply_confidence(name, accept_threshold=0.90) == ConfidencePolicy.BLOCK

    def test_unknown_always_blocks(self) -> None:
        name = RecognizedName(
            region_id="offer2", raw_text="Qliphoth", normalized="qliphoth",
            confidence=0.98, method=MatchMethod.FUZZY, matched_card_id=None,
            needs_dataset_entry=True,
        )
        assert apply_confidence(name, accept_threshold=0.90) == ConfidencePolicy.BLOCK

    def test_exact_match_ignores_threshold(self) -> None:
        name = RecognizedName(
            region_id="offer1", raw_text="Carnage", normalized="carnage",
            confidence=0.50, method=MatchMethod.EXACT, matched_card_id="carnage",
        )
        assert apply_confidence(name, accept_threshold=0.99) == ConfidencePolicy.TENTATIVE


class TestResolveName:
    def test_resolve_name_builds_record_and_policy(self) -> None:
        matcher = NameMatcher(_catalogue())
        name, policy = resolve_name(matcher, "offer1", "Inf lame", 0.96, 0.90)
        assert isinstance(name, RecognizedName)
        assert name.matched_card_id == "inflame"
        assert not name.needs_dataset_entry
        assert policy == ConfidencePolicy.TENTATIVE

    def test_resolve_name_flags_unknown(self) -> None:
        matcher = NameMatcher(_catalogue())
        name, policy = resolve_name(matcher, "offer2", "Qliphoth", 0.98, 0.90)
        assert name.needs_dataset_entry is True
        assert name.matched_card_id is None
        assert policy == ConfidencePolicy.BLOCK
