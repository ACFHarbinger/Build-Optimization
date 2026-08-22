"""Decoupled service layer for Slay the Spire 2 card-reward evaluation (SA6)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.item import Item
from core.mc_planner import MCConfig, MonteCarloPlanner
from core.reward_eval import AdvisorPreferences, RunContext, evaluate_reward
from core.scoring import ScoringConfig
from core.synergy import SynergyEngine
from data.datasets.games.file_source import FileSource
from pipeline.decks.advisor_schema import Sts2AdvisorRequest, parse_request, response_to_dict

DEFAULT_CATALOGUE_PATH = Path(__file__).resolve().parents[1] / "data/sample/slay_the_spire_2_ironclad.json"


class STS2AdvisorService:
    """Stateless evaluation service for STS2 card reward choices and projected-run planning.

    Decoupled from CLI stdin/stdout so it can be instantiated in memory, tested directly,
    or wired into a local daemon / socket bridge without altering evaluation logic.
    """

    def __init__(
        self,
        catalogue_path: Optional[Path | str] = None,
        catalogue_version: str = "sample-ironclad-v1",
    ) -> None:
        self.catalogue_path = Path(catalogue_path) if catalogue_path else DEFAULT_CATALOGUE_PATH
        self.catalogue_version = catalogue_version
        self._cached_catalogue: Optional[tuple[List[Item], SynergyEngine, Dict[str, Item]]] = None

    def _load_catalogue(self) -> tuple[List[Item], SynergyEngine, Dict[str, Item]]:
        if self._cached_catalogue is None:
            source = FileSource(str(self.catalogue_path))
            items = source.fetch_items()
            synergies = SynergyEngine(source.fetch_synergies())
            index = {str(card.item_id).lower(): card for card in items}
            self._cached_catalogue = (items, synergies, index)
        return self._cached_catalogue

    def _resolve(self, card_ids: Sequence[str], index: Dict[str, Item]) -> List[Item]:
        missing = [card_id for card_id in card_ids if card_id.lower() not in index]
        if missing:
            raise ValueError("needs_dataset_entry: " + ", ".join(missing))
        return [index[card_id.lower()] for card_id in card_ids]

    def evaluate_request(self, request: Sts2AdvisorRequest) -> Dict[str, Any]:
        """Evaluate a validated typed request and return the versioned public result dictionary."""
        if request.character != "ironclad":
            raise ValueError("blocked: only the ironclad scoring pack is available")

        catalogue, synergies, index = self._load_catalogue()
        base_ids = [entry.card_id for entry in request.deck for _ in range(entry.count)]
        base = self._resolve(base_ids, index)
        offers = self._resolve(request.offers, index)

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
                "catalogue_version": self.catalogue_version,
            }
        )

    def evaluate_payload(self, payload: Any) -> Dict[str, Any]:
        """Decode/validate a raw dictionary payload and evaluate it."""
        request = parse_request(payload)
        return self.evaluate_request(request)
