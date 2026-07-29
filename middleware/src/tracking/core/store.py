"""SQLite storage backend for the WSTracker.

Uses WAL journal mode so multiple processes can read/write concurrently
without corrupting the database.  Every public method opens its own
short-lived connection and closes it before returning, which is the
simplest safe pattern for multi-process SQLite access.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple


def _safe_json_dumps(value: Any) -> str:
    """JSON-serialise *value*, falling back to ``str()`` for non-serialisable types."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    tags        TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT    PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    name          TEXT,
    status        TEXT    NOT NULL DEFAULT 'running',
    run_type      TEXT    NOT NULL DEFAULT 'generic',
    start_time    TEXT    NOT NULL,
    end_time      TEXT,
    artifact_dir  TEXT    DEFAULT '',
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS run_tags (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT    NOT NULL,
    key       TEXT    NOT NULL,
    value     REAL    NOT NULL,
    step      INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics (run_id, key);

CREATE TABLE IF NOT EXISTS artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    path          TEXT    NOT NULL,
    artifact_type TEXT    DEFAULT 'file',
    file_hash     TEXT,
    size_bytes    INTEGER,
    created_at    TEXT    NOT NULL,
    metadata      TEXT    DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS dataset_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    file_path   TEXT,
    file_hash   TEXT,
    prev_hash   TEXT,
    size_bytes  INTEGER,
    shape       TEXT,
    metadata    TEXT    DEFAULT '{}',
    timestamp   TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TrackingStore:
    """Persistent SQLite store for experiment tracking data.

    Thread-safe and multi-process safe via SQLite WAL mode.
    Each operation acquires and releases its own connection.

    Args:
        db_path: Absolute path to the ``.db`` file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._apply_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a short-lived SQLite connection with autocommit on success."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous  = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _apply_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            with suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE dataset_events ADD COLUMN shape TEXT;")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def get_or_create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return the id of an existing experiment or create a new one."""
        tags_json = json.dumps(tags or {})
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments (name, created_at, description, tags) VALUES (?, ?, ?, ?)",
                (name, self._now(), description, tags_json),
            )
            row = conn.execute("SELECT id FROM experiments WHERE name = ?", (name,)).fetchone()
            return int(row["id"])

    def list_experiments(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        experiment_id: int,
        name: Optional[str],
        run_type: str,
        artifact_dir: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO runs (id, experiment_id, name, status, run_type, start_time, artifact_dir)
                   VALUES (?, ?, ?, 'running', ?, ?, ?)""",
                (run_id, experiment_id, name, run_type, self._now(), artifact_dir),
            )

    def finish_run(
        self,
        run_id: str,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, end_time = ?, error_message = ? WHERE id = ?",
                (status, self._now(), error, run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(
        self,
        experiment_id: Optional[int] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM runs WHERE 1=1"
        args: List[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            args.append(experiment_id)
        if run_type is not None:
            query += " AND run_type = ?"
            args.append(run_type)
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        query += " ORDER BY start_time DESC"
        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_tags (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, value),
            )

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO run_tags (run_id, key, value) VALUES (?, ?, ?)",
                [(run_id, k, str(v)) for k, v in tags.items()],
            )

    def get_tags(self, run_id: str) -> Dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM run_tags WHERE run_id = ?", (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, _safe_json_dumps(value)),
            )

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()],
            )

    def get_params(self, run_id: str) -> Dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM params WHERE run_id = ?", (run_id,)).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, key, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                (run_id, key, float(value), step, self._now()),
            )

    def log_metrics_batch(
        self,
        run_id: str,
        metrics: List[Tuple[str, float, int]],
    ) -> None:
        """Insert a batch of ``(key, value, step)`` tuples in one transaction."""
        now = self._now()
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO metrics (run_id, key, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                [(run_id, k, float(v), s, now) for k, v, s in metrics],
            )

    def get_metric_history(self, run_id: str, key: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT value, step, timestamp FROM metrics WHERE run_id = ? AND key = ? ORDER BY step",
                (run_id, key),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_metrics(self, run_id: str) -> Dict[str, float]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT key, value FROM metrics
                   WHERE id IN (SELECT MAX(id) FROM metrics WHERE run_id = ? GROUP BY key)""",
                (run_id,),
            ).fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        file_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn() as conn:
            # Check if artifact already exists to avoid duplicates
            existing = conn.execute(
                "SELECT id FROM artifacts WHERE run_id = ? AND path = ?",
                (run_id, path),
            ).fetchone()

            if existing:
                return

            conn.execute(
                """INSERT INTO artifacts
                   (run_id, name, path, artifact_type, file_hash, size_bytes, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    name,
                    path,
                    artifact_type,
                    file_hash,
                    size_bytes,
                    self._now(),
                    json.dumps(metadata or {}),
                ),
            )

    def get_artifacts(
        self,
        run_id: str,
        artifact_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM artifacts WHERE run_id = ?"
        args: List[Any] = [run_id]
        if artifact_type:
            query += " AND artifact_type = ?"
            args.append(artifact_type)
        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Dataset events
    # ------------------------------------------------------------------

    def log_dataset_event(
        self,
        run_id: str,
        event_type: str,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        prev_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO dataset_events
                   (run_id, event_type, file_path, file_hash, prev_hash,
                    size_bytes, shape, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    event_type,
                    file_path,
                    file_hash,
                    prev_hash,
                    size_bytes,
                    str(tuple(shape)) if shape is not None else None,
                    json.dumps(metadata or {}),
                    self._now(),
                ),
            )

    def get_dataset_events(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_events WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
