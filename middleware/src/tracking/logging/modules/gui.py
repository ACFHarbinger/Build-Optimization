"""
GUI communication for simulation logs and daily stats.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from omegaconf import OmegaConf
from utils.configs.setup_utils import deep_sanitize

import constants as udef


def send_daily_output_to_gui(
    daily_log: Dict[str, Any],
    policy: str,
    sample_idx: int,
    day: int,
    bins_c: Sequence[float],
    collected: Sequence[float],
    bins_real_c_after: Sequence[float],
    log_path: str,
    tour: Sequence[int],
    coordinates: Union[pd.DataFrame, List[Any]],
    lock: Optional[threading.Lock] = None,
    must_go: Optional[Sequence[int]] = None,
) -> None:
    """Write daily simulation output to a log file for GUI consumption."""
    # Preserve the entire daily_log, including its pre-mapped Dataset ID "tour" list
    full_payload = {k: v for k, v in daily_log.items() if k in udef.DAY_METRICS}

    coords_lookup = None
    if isinstance(coordinates, pd.DataFrame):
        coords_lookup = coordinates.copy()
        coords_lookup.columns = [str(c).upper().strip() for c in coords_lookup.columns]

    # Output full coordinates map only on day 1 to act as a structural lookup in the UI
    if day == 1:
        route_coords = []

        node_count = len(coords_lookup) if coords_lookup is not None else len(coordinates)
        for i in range(node_count):
            point_data = _process_tour_point(i, coords_lookup)
            route_coords.append(point_data)

        full_payload.update({"all_bin_coords": route_coords})
    full_payload.update(
        {
            "bin_state_c": list(bins_c),
            "bin_state_collected": list(collected),
            "bins_state_real_c_after": list(bins_real_c_after),
        }
    )

    # Tour bin indices: 0-indexed bin IDs visited in route order (excluding depot)
    tour_indices = [int(idx) - 1 for idx in tour if int(idx) > 0]
    full_payload["tour_indices"] = tour_indices

    if must_go is not None:
        # Map 1-based bin IDs (logic) to 0-based GUI IDs (indices)
        # Bins are i=1..N, Depot is i=0 (skipped)
        mapped_must_go = [int(i) - 1 for i in must_go if int(i) > 0]
        full_payload.update({"must_go": mapped_must_go})

    full_payload = deep_sanitize(full_payload)
    converted = OmegaConf.to_container(OmegaConf.create(full_payload), resolve=True)
    if not isinstance(converted, dict):
        raise TypeError(f"GUI daily payload must be a mapping, got {type(converted)!r}")
    full_payload = {str(k): v for k, v in converted.items()}
    log_msg = f"GUI_DAY_LOG_START:{policy},{sample_idx},{day},{json.dumps(full_payload)}"
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()


def send_final_output_to_gui(
    log: Dict[str, Any],
    log_std: Optional[Dict[str, Any]],
    n_samples: int,
    policies: List[str],
    log_path: str,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Write final simulation summary to a log file for GUI consumption."""
    lgsd = (
        {k: [[0] * len(v) if isinstance(v, (tuple, list)) else 0 for v in pol_data] for k, pol_data in log.items()}
        if log_std is None
        else {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log_std.items()}
    )
    summary_data: Dict[str, Any] = {
        "log": {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log.items()},
        "log_std": lgsd,
        "n_samples": n_samples,
        "policies": list(policies),
    }

    # Deep sanitize summary_data to handle any remaining ListConfigs/DictConfigs and NumPy types
    summary_data = deep_sanitize(summary_data)
    converted_summary = OmegaConf.to_container(OmegaConf.create(summary_data), resolve=True)
    if not isinstance(converted_summary, dict):
        raise TypeError(f"GUI summary payload must be a mapping, got {type(converted_summary)!r}")
    summary_data = {str(k): v for k, v in converted_summary.items()}

    summary_message = f"GUI_SUMMARY_LOG_START: {json.dumps(summary_data)}"
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(summary_message + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write summary to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()


def _get_lat_lon(row: pd.Series) -> tuple:
    """Extract lat/lon from a row with flexible column naming."""
    lat = row.get("LATITUDE") or row.get("LAT") or row.get("Y")
    lon = row.get("LONGITUDE") or row.get("LNG") or row.get("LON") or row.get("X")
    return lat, lon


def _process_tour_point(node_idx: int, coords_lookup: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Build a tour point dict from a node index.

    Standard mapping:
    - Node 0 (Depot) -> Payload ID -1
    - Nodes 1..N (Bins) -> Payload ID 0..N-1
    """
    try:
        node_idx_int = int(node_idx)
    except (ValueError, TypeError):
        return {"id": node_idx}

    gui_id = node_idx_int - 1  # Default fallback
    dataset_id = gui_id
    if coords_lookup is not None and 0 <= node_idx_int < len(coords_lookup):
        row = coords_lookup.iloc[node_idx_int]
        dataset_id = int(row.get("ID", gui_id))

    if node_idx_int == 0:
        gui_id = -1
        p_type = "depot"
        p_label = "Depot"
    else:
        gui_id = node_idx_int - 1
        p_type = "bin"
        p_label = f"Bin {gui_id}"

    point_data: Dict[str, Any] = {"id": gui_id, "type": p_type, "dataset_id": dataset_id}

    if coords_lookup is not None and 0 <= node_idx_int < len(coords_lookup):
        row = coords_lookup.iloc[node_idx_int]
        lat, lon = _get_lat_lon(row)
        if lat is not None and lon is not None:
            point_data["lat"] = float(lat)
            point_data["lng"] = float(lon)
            point_data["popup"] = f"<b>{p_label}</b><br>Dataset ID: {dataset_id}"

    return point_data
