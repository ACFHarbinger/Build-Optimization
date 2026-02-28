"""
Build-Optimization: Hydra entry point.

Usage:
    python main.py solver=sa game=rpg pipeline=file
    python main.py solver=ga game=rpg optimization.budget=2000
    python main.py solver=alns game=moba optimization.time_limit=120
"""

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from core.scoring import ScoringConfig
from core.synergy import SynergyEngine, SynergyRule
from data.datasets.games.file_source import FileSource
from data.transforms import deduplicate, filter_by_level

logger = logging.getLogger(__name__)


def _build_scoring_config(cfg: DictConfig) -> ScoringConfig:
    """Build ScoringConfig from Hydra game config."""
    scoring = cfg.game.get("scoring", {})
    return ScoringConfig(
        stat_weights=dict(scoring.get("stat_weights", {})),
        synergy_multiplier=scoring.get("synergy_multiplier", 1.5),
        cost_penalty=scoring.get("cost_penalty", 0.0),
        slot_bonus=scoring.get("slot_bonus", 5.0),
        rarity_bonus=scoring.get("rarity_bonus", 2.0),
        diminishing_returns=scoring.get("diminishing_returns", True),
        diminishing_threshold=scoring.get("diminishing_threshold", 200.0),
    )


def _build_synergy_engine(synergy_rules: list) -> SynergyEngine:
    """Build SynergyEngine from parsed synergy rules."""
    rules = [
        SynergyRule(
            name=s.get("name", ""),
            tag=s.get("tag", ""),
            threshold=s.get("threshold", 2),
            bonus_stats=dict(s.get("bonus_stats", {})),
            description=s.get("description", ""),
        )
        for s in synergy_rules
    ]
    return SynergyEngine(rules=rules)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for build optimization."""
    print("=" * 60)
    print("  BUILD OPTIMIZATION — Videogame Build Optimizer")
    print("=" * 60)
    print(f"\n  Game:   {cfg.game.get('name', 'Unknown')}")
    print(f"  Solver: {list(cfg.solver.keys())[0] if cfg.solver else 'unknown'}")
    print(f"  Budget: {cfg.optimization.budget}")
    print(f"  Level:  {cfg.optimization.character_level}")
    print(f"  Time:   {cfg.optimization.time_limit}s")
    print()

    # 1. Load data via pipeline
    items_path = cfg.game.data.items_path
    source = FileSource(items_path=items_path)
    items = source.fetch_items()
    raw_synergies = source.fetch_synergies()

    # 2. Apply transforms
    items = filter_by_level(items, cfg.optimization.character_level)
    items = deduplicate(items)

    print(f"  Loaded {len(items)} items, {len(raw_synergies)} synergy rules")

    # 3. Build scoring and synergy engine
    _scoring_config = _build_scoring_config(cfg)
    synergy_engine = _build_synergy_engine(
        [
            {
                "name": s.name,
                "tag": s.tag,
                "threshold": s.threshold,
                "bonus_stats": s.bonus_stats,
                "description": s.description,
            }
            for s in raw_synergies
        ]
    )

    # Solver registry parsing handled by PolicyFactory
    from policies import create_policy

    solver_name = list(cfg.solver.keys())[0]

    solver_params = OmegaConf.to_container(cfg.solver[solver_name], resolve=True)
    solver_kwargs = solver_params if isinstance(solver_params, dict) else {}

    try:
        policy = create_policy(solver_name, config=solver_kwargs)
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        return

    print(f"\n  Running {solver_name.upper()}...")
    start = time.time()

    from core.problem import BuildProblem

    class DummyItemContext:
        def __init__(self, items):
            import numpy as np

            self.items = np.zeros((len(items), 5))
            self.costs = items.cost.to_numpy()

    ctx = DummyItemContext(items)
    problem = BuildProblem(ctx.items, ctx.costs)

    best_build, best_score, _ = policy.execute(
        problem=problem,
        budget=cfg.optimization.budget,
    )

    elapsed = time.time() - start

    # 5. Display results
    print(f"\n{'=' * 60}")
    print(f"  RESULT — {solver_name.upper()} ({elapsed:.2f}s)")
    print(f"{'=' * 60}")
    print(f"\n  Score: {best_score:.2f}")
    print(f"  Cost:  {best_build.total_cost:.0f} / {cfg.optimization.budget:.0f}")
    print(f"  Items: {len(best_build.equipped_items)} / {len(best_build.slots)}")
    print()

    for slot, item in best_build.slots.items():
        if item is not None:
            stats_str = ", ".join(f"{k}={v:.0f}" for k, v in item.stats.items())
            print(f"    [{slot.name:>12}] {item.name:<25} ({item.rarity.name}, ${item.cost:.0f}) — {stats_str}")
        else:
            print(f"    [{slot.name:>12}] (empty)")

    # Show aggregate stats
    print("\n  Total Stats:")
    for stat, value in sorted(best_build.total_stats.items()):
        print(f"    {stat:<20} {value:>8.1f}")

    # Show active synergies
    if cfg.output.get("show_synergies", True) and synergy_engine:
        active = synergy_engine.active_synergies(best_build)
        if active:
            print("\n  Active Synergies:")
            for syn in active:
                print(f"    ✦ {syn}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
