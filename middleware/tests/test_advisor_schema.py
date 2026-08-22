"""Contract and CLI tests for SA1/SA6."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor_cli import evaluate_payload  # noqa: E402

from pipeline.decks.advisor_schema import CONTRACT_VERSION, parse_request


def _payload() -> dict:
    return {
        "character": "ironclad",
        "deck": [{"card_id": "strike", "count": 5}, {"card_id": "defend", "count": 4}, {"card_id": "bash", "count": 1}],
        "offers": ["carnage", "cleave", "inflame"],
        "context": {"act": 1, "floor": 6, "hp_pct": 0.8, "gold": 100},
        "preferences": {"mc_rollouts": 3, "seed": 7},
    }


def test_parse_request_defaults_contract_version() -> None:
    request = parse_request(_payload())
    assert request.contract_version == CONTRACT_VERSION
    assert sum(entry.count for entry in request.deck) == 10


def test_parse_request_requires_three_offers() -> None:
    payload = _payload()
    payload["offers"] = ["carnage"]
    try:
        parse_request(payload)
    except ValueError as exc:
        assert "exactly three" in str(exc)
    else:
        raise AssertionError("invalid offer count was accepted")


def test_cli_evaluates_known_cards_and_returns_contract() -> None:
    response = evaluate_payload(_payload())
    assert response["status"] == "ok"
    assert response["contract_version"] == CONTRACT_VERSION
    assert len(response["choices"]) == 4
    assert response["choices"][0]["metrics"]["mc_projected_mean"] >= 0


def test_cli_blocks_unknown_card_without_guessing() -> None:
    payload = _payload()
    payload["offers"][0] = "unknown card"
    try:
        evaluate_payload(payload)
    except ValueError as exc:
        assert "needs_dataset_entry" in str(exc)
    else:
        raise AssertionError("unknown card was accepted")
