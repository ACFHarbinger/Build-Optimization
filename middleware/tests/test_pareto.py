"""Tests for core.pareto — non-dominated set over mixed max/min objectives (SA4)."""

from __future__ import annotations

from core.pareto import ADVISOR_OBJECTIVES, ObjectiveSpec, dominates, non_dominated_indices


class TestDominates:
    def test_strictly_better_on_one_and_equal_elsewhere(self) -> None:
        specs = (ObjectiveSpec("a", True), ObjectiveSpec("b", True))
        assert dominates({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, specs)
        assert not dominates({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0}, specs)

    def test_equal_points_do_not_dominate(self) -> None:
        specs = (ObjectiveSpec("a", True),)
        point = {"a": 1.0}
        assert not dominates(point, point, specs)

    def test_minimise_objective_treats_smaller_as_better(self) -> None:
        specs = (ObjectiveSpec("dilution", maximize=False), ObjectiveSpec("tempo", maximize=True))
        skip = {"dilution": 0.0, "tempo": 5.0}
        take = {"dilution": 2.0, "tempo": 5.0}
        assert dominates(skip, take, specs)
        assert not dominates(take, skip, specs)

    def test_tradeoff_neither_dominates(self) -> None:
        specs = ADVISOR_OBJECTIVES
        skip = {"tempo": 4.0, "synergy": 0.0, "dilution": 0.0, "resilience": 2.0}
        take = {"tempo": 6.0, "synergy": 1.0, "dilution": 1.5, "resilience": 1.5}
        assert not dominates(skip, take, specs)
        assert not dominates(take, skip, specs)


class TestNonDominated:
    def test_empty(self) -> None:
        assert non_dominated_indices([]) == []

    def test_retains_tradeoff_pair(self) -> None:
        specs = ADVISOR_OBJECTIVES
        points = [
            {"tempo": 4.0, "synergy": 0.0, "dilution": 0.0, "resilience": 2.0},  # skip
            {"tempo": 6.0, "synergy": 1.0, "dilution": 1.5, "resilience": 1.5},  # take
            {"tempo": 3.0, "synergy": 0.0, "dilution": 2.0, "resilience": 1.0},  # dominated junk
        ]
        kept = non_dominated_indices(points, specs)
        assert kept == [0, 1]

    def test_equal_points_all_kept(self) -> None:
        specs = (ObjectiveSpec("a", True),)
        points = [{"a": 1.0}, {"a": 1.0}]
        assert non_dominated_indices(points, specs) == [0, 1]

    def test_single_point(self) -> None:
        assert non_dominated_indices([{"tempo": 1.0, "synergy": 0.0, "dilution": 0.0, "resilience": 1.0}]) == [0]
