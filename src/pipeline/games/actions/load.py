"""
LoadItemsAction: loads items and synergies from the configured data source.

Used by LoadingState (and usable standalone via the Command Pattern interface)
to fetch, filter, and pre-process game data from JSON or CSV files.
"""

from typing import Any, Dict

from core.synergy import SynergyEngine
from data.datasets.games.file_source import FileSource
from data.transforms import deduplicate, filter_by_level

from ..problem import BuildProblem
from .base import GameAction


class LoadItemsAction(GameAction):
    """
    Loads items and synergies; constructs the SynergyEngine and BuildProblem.

    Expected context keys (read):
        ``items_path``       – Path to the game data file.
        ``character_level``  – Maximum item level to include (default 99).
        ``budget``           – Budget for the BuildProblem.
        ``scoring_config``   – ScoringConfig (optional).

    Context keys written:
        ``items``            – Filtered List[Item].
        ``synergy_engine``   – SynergyEngine.
        ``problem``          – BuildProblem instance.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        items_path: str = context["items_path"]
        character_level: int = context.get("character_level", 99)
        budget: float = context.get("budget", float("inf"))
        scoring_config = context.get("scoring_config")

        source = FileSource(items_path=items_path)
        raw_items = source.fetch_items()
        raw_synergies = source.fetch_synergies()

        items = filter_by_level(raw_items, character_level)
        items = deduplicate(items)
        context["items"] = items

        context["synergy_engine"] = SynergyEngine(rules=raw_synergies)

        context["problem"] = BuildProblem.from_items(
            items=items,
            budget=budget,
            scoring_config=scoring_config,
            synergy_engine=context["synergy_engine"],
        )
