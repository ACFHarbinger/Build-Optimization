"""Unit tests for decoupled STS2AdvisorService (SA6)."""

from __future__ import annotations

import pytest

from core.advisor_service import STS2AdvisorService
from pipeline.decks.advisor_schema import CardEntry, RunContextInput, Sts2AdvisorRequest


def _sample_request() -> Sts2AdvisorRequest:
    return Sts2AdvisorRequest(
        character="ironclad",
        deck=[
            CardEntry(card_id="strike", count=5),
            CardEntry(card_id="defend", count=4),
            CardEntry(card_id="bash", count=1),
        ],
        offers=["carnage", "cleave", "inflame"],
        context=RunContextInput(act=1, floor=6, hp_pct=0.8, gold=100),
    )


def test_service_initialization_and_evaluation() -> None:
    service = STS2AdvisorService()
    request = _sample_request()
    response = service.evaluate_request(request)

    assert response["status"] == "ok"
    assert response["character"] == "ironclad"
    assert response["base_deck_size"] == 10
    assert len(response["choices"]) == 4
    assert response["recommendation"] in ["skip", "carnage", "cleave", "inflame"]
    assert len(response["pareto_front"]) >= 1

    # Check metrics structure
    for choice in response["choices"]:
        metrics = choice["metrics"]
        assert "tempo_score" in metrics
        assert "synergy_score" in metrics
        assert "dilution_penalty" in metrics
        assert "mc_projected_mean" in metrics
        assert "mc_projected_ci_lower" in metrics
        assert "mc_projected_ci_upper" in metrics
        assert metrics["mc_projected_ci_lower"] <= metrics["mc_projected_ci_upper"]


def test_service_evaluates_raw_payload() -> None:
    service = STS2AdvisorService()
    payload = {
        "character": "ironclad",
        "deck": [{"card_id": "strike", "count": 5}, {"card_id": "defend", "count": 4}, {"card_id": "bash", "count": 1}],
        "offers": ["carnage", "cleave", "inflame"],
        "context": {"act": 1, "floor": 6},
    }
    response = service.evaluate_payload(payload)
    assert response["status"] == "ok"
    assert response["base_deck_size"] == 10


def test_service_blocks_unsupported_character() -> None:
    service = STS2AdvisorService()
    request = Sts2AdvisorRequest(
        character="silent",
        deck=[CardEntry(card_id="strike", count=5)],
        offers=["carnage", "cleave", "inflame"],
    )
    with pytest.raises(ValueError, match="only the ironclad scoring pack is available"):
        service.evaluate_request(request)


def test_service_blocks_unknown_card() -> None:
    service = STS2AdvisorService()
    request = Sts2AdvisorRequest(
        character="ironclad",
        deck=[CardEntry(card_id="nonexistent_card", count=1)],
        offers=["carnage", "cleave", "inflame"],
    )
    with pytest.raises(ValueError, match="needs_dataset_entry: nonexistent_card"):
        service.evaluate_request(request)
