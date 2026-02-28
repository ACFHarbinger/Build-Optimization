"""
LoadingState: loads game data and builds the optimization problem.

Fetches items and synergies from the configured data source, applies
transforms, constructs the SynergyEngine, and creates the BuildProblem
instance ready for the solver.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.problem import BuildProblem
from core.synergy import SynergyEngine
from data.datasets.games.file_source import FileSource
from data.transforms import deduplicate, filter_by_level

from .base import GameState

if TYPE_CHECKING:
    from ..context import GameOptimizationContext

logger = logging.getLogger(__name__)


class LoadingState(GameState):
    """Loads items, synergies, and constructs the BuildProblem."""

    def handle(self, ctx: "GameOptimizationContext") -> None:
        logger.info("Loading game data from %s", ctx.items_path)

        source = FileSource(items_path=ctx.items_path)
        raw_items = source.fetch_items()
        raw_synergies = source.fetch_synergies()

        # Apply standard transforms
        items = filter_by_level(raw_items, ctx.character_level)
        items = deduplicate(items)
        ctx.items = items

        logger.info(
            "Loaded %d items (from %d raw), %d synergy rules",
            len(items),
            len(raw_items),
            len(raw_synergies),
        )

        # Build synergy engine directly from SynergyRule objects
        ctx.synergy_engine = SynergyEngine(rules=raw_synergies)

        # Construct the BuildProblem
        ctx.problem = BuildProblem.from_items(
            items=items,
            budget=ctx.budget,
            scoring_config=ctx.scoring_config,
            synergy_engine=ctx.synergy_engine,
        )

        logger.info(
            "BuildProblem created: %d items across %d slots",
            ctx.problem.num_items,
            ctx.problem.num_slots,
        )

        from .solving import SolvingState

        ctx.transition_to(SolvingState())
