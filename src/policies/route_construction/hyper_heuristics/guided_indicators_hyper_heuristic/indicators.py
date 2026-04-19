"""
Guidance Indicators for GIHH.

This module implements the two guidance indicators used by the hyper-heuristic
to evaluate Low-Level Heuristics (LLHs) based on their Pareto performance:

1. ScoreA (Quality Reward): Increased when an offspring is accepted into the Pareto Archive.
2. ScoreB (Directional Reward): Tracks objective bias (e.g., +1 for Revenue improvement, -1 for Cost improvement).

Reference:
    Chen, B., Qu, R., Bai, R., & Laesanklang, W. (2018). "A hyper-heuristic with
    two guidance indicators for bi-objective mixed-shift vehicle routing problem
    with time windows." European Journal of Operational Research, 269(2), 661-675.
"""

from typing import Dict


class ScoreAIndicator:
    """
    ScoreA (Quality Reward).

    Measures the frequency an operator successfully produces a solution that
    is good enough to be accepted into the non-dominated Pareto Archive (ARCH).
    """

    def __init__(self) -> None:
        """Initialize ScoreA tracking."""
        self.scores: Dict[str, float] = {}

    def update(self, operator: str, accepted: bool) -> None:
        """
        Update ScoreA. Adds 1 if the solution was accepted.

        Args:
            operator: Name of the operator.
            accepted: True if the solution was accepted into ARCH.
        """
        if operator not in self.scores:
            self.scores[operator] = 0.0
        if accepted:
            self.scores[operator] += 1.0

    def get_score(self, operator: str) -> float:
        """Get the accumulated ScoreA for an operator."""
        return self.scores.get(operator, 0.0)

    def reset(self, operator: str) -> None:
        """Reset ScoreA for a specific operator (used at segment boundaries)."""
        self.scores[operator] = 0.0


class ScoreBIndicator:
    """
    ScoreB (Directional Reward).

    Measures the objective-space directional bias of an operator.
    If it improves Revenue (Objective 1), score increases by 1.
    If it improves Cost (Objective 2), score decreases by 1.
    """

    def __init__(self) -> None:
        """Initialize ScoreB tracking."""
        self.scores: Dict[str, float] = {}

    def update(self, operator: str, revenue_improved: bool, cost_improved: bool) -> None:
        """
        Update ScoreB based on objective improvements.

        [THEORETICAL CORRECTION]
        Modified from Chen et al. (2018) to avoid zero-sum neutrality.
        If a heuristic improves BOTH Revenue and Cost simultaneously (a strictly
        dominating move), it is rewarded positively (+1.0) because it represents
        ideal convergence toward the Pareto front. Standard +1/-1 logic applies
        only if exactly one objective was improved.

        Args:
            operator: Name of the operator.
            revenue_improved: True if revenue > parent's revenue.
            cost_improved: True if cost < parent's cost.
        """
        if operator not in self.scores:
            self.scores[operator] = 0.0

        if revenue_improved and cost_improved:
            # Dual-improvement: Highly favorable move
            self.scores[operator] += 1.0
        elif revenue_improved:
            self.scores[operator] += 1.0
        elif cost_improved:
            self.scores[operator] -= 1.0

    def get_score(self, operator: str) -> float:
        """Get the accumulated ScoreB for an operator."""
        return self.scores.get(operator, 0.0)

    def reset(self, operator: str) -> None:
        """Reset ScoreB for a specific operator (used at segment boundaries)."""
        self.scores[operator] = 0.0
