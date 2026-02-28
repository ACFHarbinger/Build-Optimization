"""
ReportingState: aggregates solver output into the final result dictionary.

This is the terminal state of the optimization state machine. It collates
all context attributes into a structured dict and sets ``ctx.result``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import GameState

if TYPE_CHECKING:
    from ..context import GameOptimizationContext

logger = logging.getLogger(__name__)


class ReportingState(GameState):
    """Collates solver output into the final result dict and terminates the machine."""

    def handle(self, ctx: "GameOptimizationContext") -> None:
        build = ctx.best_build
        synergy_engine = ctx.synergy_engine

        active_synergies: list = []
        if synergy_engine is not None and build is not None:
            active_synergies = synergy_engine.active_synergies(build)

        ctx.result = {
            "success": True,
            "solver": ctx.solver_name,
            "score": ctx.best_score,
            "cost": build.total_cost if build else 0.0,
            "budget": ctx.budget,
            "items_equipped": len(build) if build else 0,
            "elapsed": ctx.elapsed,
            "active_synergies": active_synergies,
            "build": build,
            "extra": ctx.extra,
            "slots": (
                {
                    slot.name: (item.name if item else None)
                    for slot, item in build.slots.items()
                }
                if build else {}
            ),
            "total_stats": build.total_stats if build else {},
        }

        logger.info(
            "Optimization complete — solver=%s score=%.4f cost=%.0f/%0.f synergies=%d",
            ctx.solver_name,
            ctx.best_score,
            ctx.result["cost"],
            ctx.budget,
            len(active_synergies),
        )

        # Terminate the state machine
        ctx.transition_to(None)
