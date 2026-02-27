"""Utilities for loading SQL files from the sql/ subdirectory.

Convention
----------
Multi-query SQL files use ``-- [section_name]`` markers to separate named
sections::

    -- [experiments]
    SELECT name FROM experiments ORDER BY created_at DESC;

    -- [runs_by_status]
    SELECT status, COUNT(*) AS count FROM runs GROUP BY status;

Single-query files (e.g. ``prune_candidates.sql``) contain no markers and
are returned as-is by :func:`load_sql`.
"""

import re
from pathlib import Path
from typing import Dict

_SQL_DIR = Path(__file__).parent / "sql"

_SECTION_RE = re.compile(r"^--\s*\[(\w+)\]\s*$", re.MULTILINE)


def load_sql(filename: str) -> str:
    """Return the raw text of *filename* from the sql/ directory."""
    return (_SQL_DIR / filename).read_text(encoding="utf-8")


def load_sections(filename: str) -> Dict[str, str]:
    """Parse a SQL file with ``-- [section_name]`` markers into an ordered dict.

    Sections appear in file order (Python 3.7+ dict ordering guarantee).
    Each body is stripped of leading/trailing whitespace.  Empty bodies are
    omitted.

    Parameters
    ----------
    filename:
        Filename relative to the ``sql/`` subdirectory, e.g. ``inspect.sql``.

    Returns
    -------
    Dict[str, str]
        Mapping of section name → SQL text (with trailing ``;`` preserved).
    """
    text = load_sql(filename)
    parts = _SECTION_RE.split(text)
    # parts layout after split: [pre_text, name1, body1, name2, body2, ...]
    sections: Dict[str, str] = {}
    it = iter(parts[1:])  # skip any text before the first marker
    for name, body in zip(it, it):
        stripped = body.strip()
        if stripped:
            sections[name] = stripped
    return sections
