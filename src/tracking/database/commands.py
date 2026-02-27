"""Database management commands for the WSmart-Route tracking store.

All SQL is loaded from the sql/ subdirectory via sql_loader.

Usage::

    python -m tracking.database.commands inspect
    python -m tracking.database.commands compact
    python -m tracking.database.commands clean
    python -m tracking.database.commands stats [--experiment NAME]
    python -m tracking.database.commands metrics [--key KEY] [--experiment NAME]
    python -m tracking.database.commands prune [--older-than N] [--status S] [--experiment E] [--dry-run]
    python -m tracking.database.commands export [--run-id UUID] [--experiment E] [--latest] [-o FILE]
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from tracking.database.cmd_stats import metrics_summary, stats_database
from tracking.database.shared import DB_PATH, _conn
from tracking.database.sql_loader import load_sections, load_sql

# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def inspect_database() -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return

    sql = load_sections("inspect.sql")
    conn = _conn()
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

    experiments = conn.execute(sql["experiments"]).fetchall()
    status_counts = {r["status"]: r["count"] for r in conn.execute(sql["runs_by_status"]).fetchall()}
    type_counts = {r["run_type"]: r["count"] for r in conn.execute(sql["runs_by_type"]).fetchall()}
    totals = conn.execute(sql["record_counts"]).fetchone()
    recent_runs = conn.execute(sql["recent_runs"]).fetchall()
    conn.close()

    print()
    print("=" * 64)
    print("  WSmart-Route Tracking Database")
    print("=" * 64)
    print(f"  File : {DB_PATH}  ({size_mb:.2f} MB)")
    print()
    print(f"  Experiments ({len(experiments)}) :")
    for exp in experiments:
        print(f"    • {exp['name']}  ({exp['created_at'][:10]})")
    print()
    print("  Runs by status :")
    for s, c in sorted(status_counts.items()):
        print(f"    • {s:<13} {c:>6,}")
    print()
    print("  Runs by type :")
    for t, c in sorted(type_counts.items()):
        print(f"    • {t:<13} {c:>6,}")
    print()
    print(f"  Metric rows      : {totals['metric_rows']:>10,}")
    print(f"  Param rows       : {totals['param_rows']:>10,}")
    print(f"  Artifact rows    : {totals['artifact_rows']:>10,}")
    print(f"  Dataset events   : {totals['dataset_event_rows']:>10,}")
    print()
    print("  Most recent runs :")
    hdr = f"    {'ID':<10} {'Type':<13} {'Status':<12} {'Experiment':<32} Started"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in recent_runs:
        print(
            f"    {r['id'][:8]:<10} {r['run_type']:<13} {r['status']:<12}"
            f" {r['experiment_name'][:32]:<32} {r['start_time'][:19]}"
        )
    print()


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def clean_database() -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        conn.executescript(load_sql("clean.sql"))
        conn.close()
        print("✅ Tracking database cleaned (data removed, schema preserved).")
    except sqlite3.OperationalError as e:
        print(f"❌ Failed to clean tracking database: {e}")


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def compact_database() -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return

    sql = load_sections("compact.sql")
    size_before = os.path.getsize(DB_PATH)
    print(f"  File        : {DB_PATH}")
    print(f"  Size before : {size_before / 1024:.1f} KB")

    conn = _conn(timeout=10.0)
    result = conn.execute(sql["integrity_check"]).fetchone()
    ok = result is not None and result[0] == "ok"
    print(f"  Integrity   : {'✅ ok' if ok else '❌  ' + str(result[0])}")

    if not ok:
        conn.close()
        print("❌ Aborting — fix corruption before compacting.")
        return

    conn.execute(sql["wal_checkpoint"])
    conn.execute(sql["vacuum"])
    conn.close()

    size_after = os.path.getsize(DB_PATH)
    saved = size_before - size_after
    print(f"  Size after  : {size_after / 1024:.1f} KB")
    if saved > 0:
        print(f"  Reclaimed   : {saved / 1024:.1f} KB")
    elif saved == 0:
        print("  Reclaimed   : 0 KB  (no fragmentation)")
    else:
        print(f"  Size change : +{-saved / 1024:.1f} KB  (WAL merged into main file)")
    print("✅ Database compacted.")


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def prune_database(
    older_than_days: int = 30,
    status: str = "failed",
    experiment_name: str = "",
    dry_run: bool = False,
) -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return

    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    candidate_sql = load_sql("prune_candidates.sql")
    delete_sql = load_sections("prune_delete.sql")

    conn = _conn()
    runs = conn.execute(
        candidate_sql,
        {"cutoff": cutoff, "status": status, "experiment_name": experiment_name},
    ).fetchall()

    if not runs:
        criteria = f"status={status!r}, older_than={older_than_days}d"
        if experiment_name:
            criteria += f", experiment={experiment_name!r}"
        print(f"ℹ️  No runs match the prune criteria ({criteria}).")
        conn.close()
        return

    verb = "Would remove" if dry_run else "Removing"
    print(f"\n{verb} {len(runs)} run(s):\n")
    print(f"  {'ID':<10} {'Status':<12} {'Type':<13} {'Experiment':<32} Started")
    print("  " + "-" * 82)
    for r in runs:
        print(
            f"  {r['id'][:8]:<10} {r['status']:<12} {r['run_type']:<13}"
            f" {r['experiment_name'][:32]:<32} {r['start_time'][:10]}"
        )

    if dry_run:
        print("\n(dry-run — no changes made)")
        conn.close()
        return

    for r in runs:
        for stmt in delete_sql.values():
            conn.execute(stmt, (r["id"],))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    print(f"\n✅ Pruned {len(runs)} run(s) and reclaimed disk space.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _resolve_run_id(conn: sqlite3.Connection, run_id: str, experiment_name: str, latest: bool) -> str:
    sql = load_sections("resolve_run.sql")
    if latest or (not run_id and experiment_name):
        row = conn.execute(sql["latest"], {"experiment_name": experiment_name}).fetchone()
        if row is None:
            print("❌ No runs found.", file=sys.stderr)
            sys.exit(1)
        return row["id"]
    if not run_id:
        print("❌ Provide --run-id, --latest, or --experiment.", file=sys.stderr)
        sys.exit(1)
    if len(run_id) < 32:
        row = conn.execute(sql["by_prefix"], {"prefix": run_id + "%"}).fetchone()
        if row is None:
            print(f"❌ No run matching prefix '{run_id}'.", file=sys.stderr)
            sys.exit(1)
        return row["id"]
    return run_id


def export_run(run_id: str = "", experiment_name: str = "", latest: bool = False, output: str = "") -> None:
    if not os.path.exists(DB_PATH):
        print("❌ Tracking database not found.", file=sys.stderr)
        sys.exit(1)

    conn = _conn()
    full_id = _resolve_run_id(conn, run_id, experiment_name, latest)
    sql = load_sections("export_run.sql")
    p = {"run_id": full_id}

    run_row = conn.execute(sql["run_detail"], p).fetchone()
    if run_row is None:
        print(f"❌ Run '{full_id}' not found.", file=sys.stderr)
        sys.exit(1)

    data: Dict[str, Any] = dict(run_row)
    data["tags"] = {r["key"]: r["value"] for r in conn.execute(sql["tags"], p).fetchall()}
    data["params"] = {}
    for r in conn.execute(sql["params"], p).fetchall():
        try:
            data["params"][r["key"]] = json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            data["params"][r["key"]] = r["value"]
    metrics: Dict[str, List[Dict[str, Any]]] = {}
    for r in conn.execute(sql["metrics"], p).fetchall():
        metrics.setdefault(r["key"], []).append({"step": r["step"], "value": r["value"], "timestamp": r["timestamp"]})
    data["metrics"] = metrics
    data["artifacts"] = [dict(r) for r in conn.execute(sql["artifacts"], p).fetchall()]
    data["dataset_events"] = [dict(r) for r in conn.execute(sql["dataset_events"], p).fetchall()]
    conn.close()

    payload = json.dumps(data, indent=2, default=str)
    if output:
        with open(output, "w") as f:
            f.write(payload)
        print(f"✅ Run {full_id[:8]} exported to {output}")
    else:
        print(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="commands",
        description="WSmart-Route tracking database management commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Print database overview and recent runs.")
    sub.add_parser("clean", help="Delete all data, preserve schema.")
    sub.add_parser("compact", help="Integrity check, WAL checkpoint, VACUUM.")

    p_stats = sub.add_parser("stats", help="Show comprehensive database statistics.")
    p_stats.add_argument("--experiment", default="", metavar="NAME", help="Limit statistics to one experiment.")

    p_metrics = sub.add_parser("metrics", help="Show per-metric statistics.")
    p_metrics.add_argument("--key", default="", metavar="KEY", help="Drill into a specific metric key.")
    p_metrics.add_argument("--experiment", default="", metavar="NAME", help="Limit to one experiment.")

    p_prune = sub.add_parser("prune", help="Remove stale runs.")
    p_prune.add_argument("--older-than", type=int, default=30, metavar="DAYS")
    p_prune.add_argument("--status", default="failed", choices=["failed", "running", "completed", "all"])
    p_prune.add_argument("--experiment", default="", metavar="NAME")
    p_prune.add_argument("--dry-run", action="store_true")

    p_export = sub.add_parser("export", help="Dump a run to JSON.")
    p_export.add_argument("--run-id", default="", metavar="UUID")
    p_export.add_argument("--experiment", default="", metavar="NAME")
    p_export.add_argument("--latest", action="store_true")
    p_export.add_argument("--output", "-o", default="", metavar="FILE")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_database()
    elif args.command == "clean":
        clean_database()
    elif args.command == "compact":
        compact_database()
    elif args.command == "prune":
        prune_database(args.older_than, args.status, args.experiment, args.dry_run)
    elif args.command == "export":
        export_run(args.run_id, args.experiment, args.latest, args.output)
    elif args.command == "stats":
        stats_database(args.experiment)
    elif args.command == "metrics":
        metrics_summary(args.key, args.experiment)


if __name__ == "__main__":
    main()
