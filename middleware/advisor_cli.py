"""One-shot stdin/stdout entry point for the STS2 reward advisor (SA6)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.mc_planner import MCConfig, MonteCarloPlanner  # noqa: E402
from core.reward_eval import AdvisorPreferences, RunContext, evaluate_reward  # noqa: E402
from core.scoring import ScoringConfig  # noqa: E402
from core.synergy import SynergyEngine  # noqa: E402
from data.datasets.games.file_source import FileSource  # noqa: E402
from pipeline.decks.advisor_schema import parse_request, response_to_dict  # noqa: E402

CATALOGUE_PATH = Path(__file__).resolve().parent / "src/data/sample/slay_the_spire_2_ironclad.json"


def _catalogue() -> tuple[List[Any], SynergyEngine]:
    source = FileSource(str(CATALOGUE_PATH))
    return source.fetch_items(), SynergyEngine(source.fetch_synergies())


def _by_id(cards: Sequence[Any]) -> Dict[str, Any]:
    return {str(card.item_id).lower(): card for card in cards}


def _resolve(card_ids: Sequence[str], index: Dict[str, Any]) -> List[Any]:
    missing = [card_id for card_id in card_ids if card_id.lower() not in index]
    if missing:
        raise ValueError("needs_dataset_entry: " + ", ".join(missing))
    return [index[card_id.lower()] for card_id in card_ids]


def evaluate_payload(payload: Any) -> Dict[str, Any]:
    """Evaluate a decoded request and return the versioned public result."""
    request = parse_request(payload)
    if request.character != "ironclad":
        raise ValueError("blocked: only the ironclad scoring pack is available")
    catalogue, synergies = _catalogue()
    index = _by_id(catalogue)
    base_ids = [entry.card_id for entry in request.deck for _ in range(entry.count)]
    base, offers = _resolve(base_ids, index), _resolve(request.offers, index)
    context = RunContext(**request.context.__dict__)
    preferences = AdvisorPreferences(
        tempo_weight=request.preferences.tempo_weight,
        synergy_weight=request.preferences.synergy_weight,
        dilution_weight=request.preferences.dilution_weight,
        resilience_weight=request.preferences.resilience_weight,
    )
    scoring = ScoringConfig(slot_bonus=0.0)
    advice = evaluate_reward(base, offers, scoring, synergies, preferences, context)
    planner = MonteCarloPlanner(
        catalogue,
        scoring,
        synergies,
        MCConfig(seed=request.preferences.seed, rollouts=request.preferences.mc_rollouts),
    )
    projections = {"skip": planner.project(base, None, context.gold)}
    projections.update({card.item_id: planner.project(base, card, context.gold) for card in offers})
    choices = []
    for action in advice.actions:
        projection = projections[action.label()]
        choices.append(
            {
                "action": action.action,
                "card_id": action.card_id,
                "card_name": action.card_name,
                "is_upgrade": action.is_upgrade,
                "rank": action.rank,
                "total_score": action.weighted_score,
                "score_delta": action.score_delta,
                "pareto_optimal": action.pareto_optimal,
                "synergy_deltas": action.synergies_gained,
                "explanation": action.explanation,
                "metrics": {
                    "tempo_score": action.objectives.tempo,
                    "synergy_score": action.objectives.synergy,
                    "dilution_penalty": action.objectives.dilution,
                    "mc_projected_mean": projection.mean,
                    "mc_projected_ci_lower": projection.ci_lower,
                    "mc_projected_ci_upper": projection.ci_upper,
                },
            }
        )
    return response_to_dict(
        {
            "status": "ok",
            "character": request.character,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "base_deck_size": len(base),
            "choices": choices,
            "pareto_front": advice.pareto_front,
            "recommendation": advice.recommendation,
            "diagnostics": "sample Ironclad catalogue; seeded projected-run bands",
            "catalogue_version": "sample-ironclad-v1",
        }
    )


def main() -> int:
    try:
        print(json.dumps(evaluate_payload(json.load(sys.stdin))))
        return 0
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps(response_to_dict({"status": "blocked", "diagnostics": str(exc)})))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
