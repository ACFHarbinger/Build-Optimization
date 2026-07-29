"""
Scoring / fitness function for videogame build optimization.

This is the objective function — analogous to the profit calculation in VRP.
It combines raw stats, synergy bonuses, and configurable weights into a
single effectiveness score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .build import Build
from .synergy import SynergyEngine


@dataclass
class ScoringConfig:
    """Configuration for how stats are weighted in the fitness function.

    Attributes:
        stat_weights: Weight per stat name (default 1.0 for unlisted stats).
        synergy_multiplier: Global multiplier for synergy bonus contribution.
        cost_penalty: Penalty per unit of cost (encourages cheaper builds).
        slot_bonus: Bonus per filled slot (encourages complete builds).
        rarity_bonus: Extra score per rarity tier of equipped items.
        diminishing_returns: If True, apply sqrt scaling to over-stacked stats.
        diminishing_threshold: Stat value above which diminishing returns kick in.
    """

    stat_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "attack": 1.0,
            "defense": 0.8,
            "speed": 0.6,
            "health": 0.7,
            "critical_rate": 1.2,
            "critical_damage": 1.0,
        }
    )
    synergy_multiplier: float = 1.5
    cost_penalty: float = 0.0
    slot_bonus: float = 5.0
    rarity_bonus: float = 2.0
    diminishing_returns: bool = True
    diminishing_threshold: float = 200.0


def score_build(
    build: Build,
    synergy_engine: Optional[SynergyEngine] = None,
    config: Optional[ScoringConfig] = None,
) -> float:
    """Compute the effectiveness score for a build.

    This is the main objective function that all solvers optimize.

    Args:
        build: The character build to score.
        synergy_engine: Optional synergy evaluator for bonus stats.
        config: Scoring weights and parameters.

    Returns:
        Float effectiveness score (higher is better).
    """
    if config is None:
        config = ScoringConfig()

    score = 0.0

    # 1. Base stat contribution (weighted)
    base_stats = build.total_stats
    for stat_name, value in base_stats.items():
        weight = config.stat_weights.get(stat_name, 1.0)

        if config.diminishing_returns and value > config.diminishing_threshold:
            # sqrt scaling above threshold
            effective = config.diminishing_threshold + (value - config.diminishing_threshold) ** 0.5
        else:
            effective = value

        score += effective * weight

    # 2. Synergy bonus contribution
    if synergy_engine is not None:
        synergy_bonuses = synergy_engine.evaluate(build)
        for stat_name, value in synergy_bonuses.items():
            weight = config.stat_weights.get(stat_name, 1.0)
            score += value * weight * config.synergy_multiplier

    # 3. Slot fill bonus (encourages complete builds)
    score += len(build.equipped_items) * config.slot_bonus

    # 4. Rarity bonus
    for item in build.equipped_items:
        score += item.rarity.multiplier * config.rarity_bonus

    # 5. Cost penalty (optional — encourages budget efficiency)
    if config.cost_penalty > 0:
        score -= build.total_cost * config.cost_penalty

    return score
