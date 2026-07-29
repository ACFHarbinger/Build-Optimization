"""
Tests for the C++ backend bridge (core.native_backend) and the "bnb"
solver it powers (pipeline.games.states.solving::_solve_bnb).

Skipped entirely if the backend extension hasn't been built (see
`backend/pixi.toml`'s `build` task) — these are integration tests against
the real compiled module, not something to fake out with a mock.
"""

import os

import pytest

from core.native_backend import load_backend

try:
    load_backend()
    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _BACKEND_AVAILABLE, reason="backend/ extension not built (pixi run build)")


class TestNativeBackend:
    def test_solve_mckp_branch_and_bound_matches_hand_computed_optimum(self) -> None:
        backend = load_backend()
        options = [
            backend.MckpOption(0, 3, 5.0),
            backend.MckpOption(0, 5, 9.0),
            backend.MckpOption(1, 4, 6.0),
            backend.MckpOption(1, 2, 3.0),
        ]
        result = backend.solve_mckp_branch_and_bound(options, 2, 7)
        assert result.total_value == 12.0
        assert result.total_weight == 7
        assert sorted(result.selected_option_indices) == [1, 3]


class TestBnbSolver:
    def test_bnb_is_at_least_as_good_as_greedy(self) -> None:
        from pipeline.games import run_optimization

        items_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample", "rpg.json")

        r_greedy = run_optimization("greedy", items_path, budget=1200.0, character_level=30, persist=False)
        r_bnb = run_optimization("bnb", items_path, budget=1200.0, character_level=30, persist=False)

        assert r_bnb["extra"]["fast_score"] >= r_greedy["extra"]["fast_score"] - 1e-6
        assert r_bnb["cost"] <= 1200.0

    def test_bnb_result_is_feasible_and_respects_slots(self) -> None:
        from pipeline.games import run_optimization

        items_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample", "rpg.json")
        result = run_optimization("bnb", items_path, budget=5000.0, character_level=30, persist=False)

        assert result["success"]
        assert result["cost"] <= 5000.0
        build = result["build"]
        for slot, item in build.slots.items():
            if item is not None:
                assert item.slot == slot
