"""
Data loading utilities for the dashboard.

Discovers solver result files and game data.
"""

import json
import os
from typing import Any, Dict, List, Optional


def _find_root() -> str:
    """Best-effort root directory discovery."""
    # Walk up from this file until we find pyproject.toml
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    return os.getcwd()


ROOT_DIR = _find_root()


def discover_solver_results(directory: Optional[str] = None) -> List[str]:
    """Find all solver result JSON files.

    Returns:
        Sorted list of absolute paths to result files.
    """
    search_dir = directory or os.path.join(ROOT_DIR, "outputs")
    if not os.path.isdir(search_dir):
        return []

    results = []
    for root, _dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".json") and ("result" in f.lower() or "solution" in f.lower()):
                results.append(os.path.join(root, f))
    return sorted(results)


def load_solver_result(path: str) -> Dict[str, Any]:
    """Load a single solver result JSON file."""
    with open(path) as fh:
        return json.load(fh)


def load_items_from_json(path: str) -> List[Dict[str, Any]]:
    """Load item data from a JSON file.

    Expects a JSON array of item dicts or a dict with an 'items' key.
    """
    with open(path) as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return []


def discover_item_files(directory: Optional[str] = None) -> List[str]:
    """Find all item data files (JSON/CSV)."""
    search_dir = directory or os.path.join(ROOT_DIR, "data")
    if not os.path.isdir(search_dir):
        return []

    results = []
    for root, _dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith((".json", ".csv")) and "item" in f.lower():
                results.append(os.path.join(root, f))
    return sorted(results)
