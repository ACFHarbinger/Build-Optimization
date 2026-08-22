"""
Tests for the SA8 local screenshot recognition pipeline.

Two tiers:
  - Pure fixture-geometry tests (CropBoxes) — always run, no OCR deps.
  - OCR transcription tests against the *committed* synthetic fixtures — gated
    on Pillow + pytesseract being importable (the graceful-skip precedent used
    for core.native_backend), since OCR is a deliberate new dependency and the
    environment may not always have them.

The repository commits the deterministic synthetic fixtures (never real game
screenshots); tests must pass here regardless of whether the OCR deps are
installed in a given environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.recognition import ConfidencePolicy, NameMatcher
from pipeline.recognition import (
    CropBoxes,
    load_image,
    ocr_available,
)
from pipeline.recognition.fixtures import RewardLayout, generate_fixtures

try:
    from pipeline.recognition import OcrTranscriber

    _OCR_DEPS = ocr_available()
except ImportError:  # pragma: no cover - guarded below
    _OCR_DEPS = False

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Subset catalogue that includes all clean fixture names + a known unknown.
_CATALOGUE = [
    ("carnage", "Carnage"),
    ("inflame", "Inflame"),
    ("cleave", "Cleave"),
    ("pommel_strike", "Pommel Strike"),
    ("demon_form", "Demon Form"),
    ("bludgeon", "Bludgeon"),
]


def _config() -> dict:
    return {
        "offers": [
            [0.06, 0.18, 0.31, 0.95],
            [0.36, 0.18, 0.61, 0.95],
            [0.66, 0.18, 0.91, 0.95],
        ],
        "name_banner": [0.10, 0.82, 0.90, 0.95],
        "deck_grid": None,
    }


class TestCropBoxes:
    def test_banner_boxes_respect_relative_layout(self) -> None:
        boxes = CropBoxes.from_config(_config())
        banner = boxes.banner_boxes(1280, 720)
        assert len(banner) == 3
        for left, top, right, bottom in banner:
            assert 0 <= left < right <= 1280
            assert 0 <= top < bottom <= 720
        # banner is a sub-region: narrower + lower than the offer region.
        offers = boxes.to_absolute(1280, 720)
        for obox, bbox in zip(offers, banner):
            assert bbox[0] >= obox[0]
            assert bbox[2] <= obox[2]
            assert bbox[1] >= obox[1]

    def test_resolution_independent_fractions(self) -> None:
        boxes = CropBoxes.from_config(_config())
        # Same fractional layout at a 16:10 size yields valid, scaled boxes.
        banner_169 = boxes.banner_boxes(1280, 720)
        banner_1610 = boxes.banner_boxes(1280, 800)
        assert len(banner_169) == len(banner_1610) == 3
        for b169, b1610 in zip(banner_169, banner_1610):
            assert b1610[3] - b1610[1] > b169[3] - b169[1]  # taller frame → taller banner


class TestFixtureGeneration:
    def test_generator_is_deterministic(self, tmp_path: Path) -> None:
        out1 = tmp_path / "a"
        out2 = tmp_path / "b"
        generate_fixtures(out1, seed=42)
        generate_fixtures(out2, seed=42)
        for name in ("reward_clean.png", "reward_degraded.png", "reward_unknown.png"):
            assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    def test_generator_writes_all_three_fixtures(self) -> None:
        out = Path(__file__).parent / "fixtures"
        for name in ("reward_clean.png", "reward_degraded.png", "reward_unknown.png"):
            assert (out / name).exists(), f"missing committed fixture {name}"

    def test_reward_layout_is_valid(self) -> None:
        assert len(RewardLayout.OFFERS) == 3
        # banner is a proper sub-box of each offer
        for bl, bt, br, bb in [RewardLayout.BANNER]:
            for _ol, _ot, _or, _ob in RewardLayout.OFFERS:
                assert 0 <= bl < br <= 1
                assert 0 <= bt < bb <= 1


@pytest.mark.skipif(not _OCR_DEPS, reason="Pillow/pytesseract not importable — OCR dep not installed")
class TestOcrTranscription:
    def test_clean_fixture_all_resolve_tentative(self) -> None:
        matcher = NameMatcher(_CATALOGUE)
        boxes = CropBoxes.from_config(_config())
        img = load_image(FIXTURES_DIR / "reward_clean.png")
        assert img is not None
        transcriber = OcrTranscriber(boxes, matcher, accept_threshold=0.90)
        results = transcriber.transcribe(img)

        assert len(results) == 3
        # All three clean offers must resolve against the catalogue.
        resolved = [card_id for name, _ in results for card_id in [name.matched_card_id] if card_id]
        assert len(resolved) == 3
        assert all(name.matched_card_id is not None and not name.needs_dataset_entry for name, _ in results)
        # Confidence policy: tentative for the accepted names.
        assert all(policy == ConfidencePolicy.TENTATIVE for _, policy in results)

    def test_unknown_fixture_blocks_unknown_card(self) -> None:
        matcher = NameMatcher(_CATALOGUE)
        boxes = CropBoxes.from_config(_config())
        img = load_image(FIXTURES_DIR / "reward_unknown.png")
        assert img is not None
        transcriber = OcrTranscriber(boxes, matcher, accept_threshold=0.90)
        results = transcriber.transcribe(img)

        # The middle offer is "Qliphoth" — not in the catalogue → must block
        # with needs_dataset_entry, never silently resolve.
        unknown_results = [r for r in results if r[0].region_id == "offer2"]
        assert unknown_results
        name, policy = unknown_results[0]
        assert name.matched_card_id is None
        assert name.needs_dataset_entry is True
        assert policy == ConfidencePolicy.BLOCK

    def test_ingest_from_bytes(self) -> None:
        data = (FIXTURES_DIR / "reward_clean.png").read_bytes()
        from pipeline.recognition import load_image_bytes

        img = load_image_bytes(data)
        assert img is not None  # bytes path works (clipboard paste analogue)
