"""
Tests for the experiment tracking database (tracking.core) and its wiring
into the games optimization pipeline (pipeline.games.optimizer).
"""

import json
import os

import pytest

from tracking.core.tracker import Tracker


class TestTrackingStore:
    def test_round_trip(self, tmp_path) -> None:
        tracker = Tracker(tracking_uri=str(tmp_path))
        with tracker.start_run("exp", run_type="test") as run:
            run.log_params({"solver": "sa", "budget": 5000})
            run.log_metric("score", 342.5, step=0)
            run.log_metric("score", 350.0, step=1)

        runs = tracker.list_runs(experiment_name="exp")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"

        history = run.get_metric_history("score")
        assert [h["value"] for h in history] == [342.5, 350.0]
        assert run.get_params() == {"solver": "sa", "budget": 5000}

    def test_failed_run_records_error(self, tmp_path) -> None:
        tracker = Tracker(tracking_uri=str(tmp_path))
        with pytest.raises(ValueError), tracker.start_run("exp", run_type="test") as run:
            raise ValueError("boom")

        stored = tracker.get_run(run.run_id)
        assert stored["status"] == "failed"
        assert "boom" in stored["error_message"]


@pytest.fixture
def _isolated_tracking(tmp_path, monkeypatch):
    """Point the tracking singleton and constants.ROOT_DIR at a scratch dir,
    resetting the module-level tracker so each test starts fresh."""
    import constants
    import tracking.core.tracker as tracker_mod

    monkeypatch.setattr(constants, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(tracker_mod, "_tracker", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestOptimizerTrackingIntegration:
    def test_run_optimization_persists_results(self, _isolated_tracking) -> None:
        from pipeline.games.optimizer import run_optimization

        tmp_path = _isolated_tracking
        items_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample", "rpg.json")

        result = run_optimization(
            solver_name="greedy",
            items_path=items_path,
            budget=2000.0,
            character_level=20,
            time_limit=5.0,
            experiment_name="pytest-experiment",
        )
        assert result["success"]

        out_dir = tmp_path / "outputs" / "pytest-experiment"
        result_files = list(out_dir.glob("*_result.json"))
        assert len(result_files) == 1
        with open(result_files[0]) as fh:
            payload = json.load(fh)
        assert payload["solver"] == "greedy"
        assert payload["score"] == result["score"]
        assert isinstance(payload["items"], list)

        db_path = tmp_path / "assets" / "tracking" / "tracking.db"
        assert db_path.exists()

    def test_run_optimization_persist_false_skips_disk_writes(self, _isolated_tracking) -> None:
        from pipeline.games.optimizer import run_optimization

        tmp_path = _isolated_tracking
        items_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample", "rpg.json")

        run_optimization(
            solver_name="greedy",
            items_path=items_path,
            budget=2000.0,
            character_level=20,
            persist=False,
        )

        assert not (tmp_path / "outputs").exists()
        assert not (tmp_path / "assets").exists()
