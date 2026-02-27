"""Shared database connection utilities for tracking database commands."""

import sqlite3

DB_PATH = "assets/tracking/tracking.db"


def _conn(timeout: float = 5.0) -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=timeout)
    c.row_factory = sqlite3.Row
    return c
