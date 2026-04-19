"""
Multi-Phase Route Improver.
"""

from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry


@RouteImproverRegistry.register("multi_phase")
class MultiPhaseRouteImprover(IRouteImprovement):
    """
    Composition route improver that runs multiple separate strategies in sequence.
    Example: Augmentation phase followed by Improvement and Polish phases.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Run multi-phase refinement on the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'phases' (List[str]).

        Returns:
            List[int]: Refined tour after all phases.

        Note:
            Requires at least 2 phases.
        """
        phase_names = (
            kwargs.get("phases", [])
            if kwargs.get("phases") is not None
            else self.config.get("phases", ["cheapest_insertion", "lkh"])
        )

        if len(phase_names) < 2:
            raise ValueError(
                f"MultiPhaseRouteImprover requires at least 2 phases, but got {len(phase_names)}. "
                f"Phases provided: {phase_names}"
            )

        metrics: ImprovementMetrics = {
            "algorithm": "MultiPhaseRouteImprover",
            "phases": [],
        }

        current_tour = tour
        for i, phase_name in enumerate(phase_names):
            p_cls = RouteImproverRegistry.get_route_improver_class(phase_name)
            if p_cls is None:
                raise ValueError(f"Unknown phase strategy at index {i}: {phase_name!r}")

            # Construct sub-processor with reference to full config dict
            p = p_cls(config=self.config)
            current_tour, p_metrics = p.process(current_tour, **kwargs)

            # Store phase-specific metrics
            metrics["phases"].append({"index": i + 1, "strategy": phase_name, "metrics": p_metrics})

        return current_tour, metrics
