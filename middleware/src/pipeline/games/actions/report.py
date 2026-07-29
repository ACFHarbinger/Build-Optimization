"""
ReportAction: collates solver output into a structured result dictionary.

Used by ReportingState (and usable standalone) to gather Build, score,
synergies, and diagnostics into the ``context["result"]`` key.
"""

from typing import Any, Dict

from .base import GameAction


class ReportAction(GameAction):
    """
    Collates solver output into ``context["result"]``.

    Expected context keys (read):
        ``best_build``       – core.build.Build object (may be None).
        ``best_score``       – Float score.
        ``synergy_engine``   – SynergyEngine (optional).
        ``solver_name``      – Algorithm identifier.
        ``budget``           – Maximum build cost.
        ``elapsed``          – Wall-clock seconds taken by the solver.
        ``extra``            – Extra diagnostics from the solver.

    Context keys written:
        ``result`` – Dict with keys: success, solver, score, cost, budget,
                     items_equipped, elapsed, active_synergies, build,
                     extra, slots, total_stats.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        build = context.get("best_build")
        synergy_engine = context.get("synergy_engine")

        active_synergies: list = []
        if synergy_engine is not None and build is not None:
            active_synergies = synergy_engine.active_synergies(build)

        context["result"] = {
            "success": True,
            "solver": context.get("solver_name", "unknown"),
            "score": context.get("best_score", 0.0),
            "cost": build.total_cost if build else 0.0,
            "budget": context.get("budget", 0.0),
            "items_equipped": len(build) if build else 0,
            "elapsed": context.get("elapsed", 0.0),
            "active_synergies": active_synergies,
            "build": build,
            "extra": context.get("extra", {}),
            "slots": (
                {
                    slot.name: (item.name if item else None)
                    for slot, item in build.slots.items()
                }
                if build else {}
            ),
            "total_stats": build.total_stats if build else {},
        }
