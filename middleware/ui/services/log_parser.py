"""
Training log discovery and parsing.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def discover_training_runs(directory: Optional[str] = None) -> List[str]:
    """Find all training run directories or log files.

    Returns:
        Sorted list of paths (directories with metrics.csv or event files).
    """
    from .data_loader import ROOT_DIR

    search_dir = directory or os.path.join(ROOT_DIR, "outputs")
    if not os.path.isdir(search_dir):
        return []

    runs: List[str] = []
    for root, _dirs, files in os.walk(search_dir):
        for f in files:
            if f in ("metrics.csv", "training_log.jsonl", "training_log.csv"):
                runs.append(root)
                break
    return sorted(runs)


def parse_training_log(run_dir: str) -> pd.DataFrame:
    """Parse a training log directory into a DataFrame.

    Supports:
    - metrics.csv (epoch, train_loss, val_loss, reward, ...)
    - training_log.jsonl (one JSON object per line)
    - training_log.csv
    """
    csv_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    jsonl_path = os.path.join(run_dir, "training_log.jsonl")
    if os.path.exists(jsonl_path):
        records: List[Dict[str, Any]] = []
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    csv2_path = os.path.join(run_dir, "training_log.csv")
    if os.path.exists(csv2_path):
        return pd.read_csv(csv2_path)

    return pd.DataFrame()
