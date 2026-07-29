"""stats and metrics subcommands for the tracking database CLI.

Invoked via commands.py; not intended to be run directly.
"""

import os

from tracking.database.shared import DB_PATH, _conn
from tracking.database.sql_loader import load_sections

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    n_f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_f) < 1024.0:
            return f"{n_f:.1f} {unit}"
        n_f /= 1024.0
    return f"{n_f:.1f} TB"


def _human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _sparkbar(value: int, max_value: int, width: int = 20) -> str:
    if max_value == 0:
        return "░" * width
    filled = round(value / max_value * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def stats_database(experiment_name: str = "") -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return

    sql = load_sections("stats.sql")
    conn = _conn()
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    p = {"experiment_name": experiment_name}

    table_sizes = conn.execute(sql["table_sizes"]).fetchall()
    exp_stats = conn.execute(sql["experiment_stats"], p).fetchall()
    top_metrics = conn.execute(sql["top_metrics"], p).fetchall()
    artifact_stats = conn.execute(sql["artifact_type_stats"], p).fetchall()
    event_stats = conn.execute(sql["dataset_event_stats"], p).fetchall()
    duration_row = conn.execute(sql["run_duration_stats"], p).fetchone()
    activity = conn.execute(sql["run_activity"], p).fetchall()
    conn.close()

    title = "WSmart-Route Tracking Database — Statistics"
    if experiment_name:
        title += f"  [{experiment_name}]"

    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)
    print(f"  File : {DB_PATH}  ({size_mb:.2f} MB)")
    print()

    # Table sizes
    print("  Table Sizes :")
    for row in table_sizes:
        print(f"    • {row['table_name']:<18} {row['rows']:>10,} rows")
    print()

    # Experiment summary
    if exp_stats:
        print("  Experiment Summary :")
        hdr = f"    {'Experiment':<34} {'Total':>6} {'Done':>6} {'Fail':>6} {'Run':>5} {'Avg Dur':>10}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for r in exp_stats:
            dur = _human_duration(r["avg_duration_s"]) if r["avg_duration_s"] is not None else "—"
            print(
                f"    {r['experiment'][:34]:<34} {r['total_runs']:>6}"
                f" {(r['completed'] or 0):>6} {(r['failed'] or 0):>6}"
                f" {(r['running'] or 0):>5} {dur:>10}"
            )
        print()

    # Run duration statistics
    if duration_row and duration_row["finished_runs"]:
        d = duration_row
        print("  Run Duration (finished runs) :")
        print(f"    • Count  : {d['finished_runs']:>8,}")
        print(f"    • Min    : {_human_duration(d['min_s']):>10}")
        print(f"    • Max    : {_human_duration(d['max_s']):>10}")
        print(f"    • Mean   : {_human_duration(d['mean_s']):>10}")
        print()

    # Top metrics
    if top_metrics:
        scope = f", experiment={experiment_name!r}" if experiment_name else ""
        print(f"  Top Metrics (by run coverage{scope}) :")
        hdr = f"    {'Key':<36} {'Runs':>5} {'Steps':>8} {'Min':>10} {'Max':>10} {'Mean':>10}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for r in top_metrics:
            print(
                f"    {r['key'][:36]:<36} {r['runs_tracking']:>5} {r['total_steps']:>8,}"
                f" {r['min_val']:>10.4f} {r['max_val']:>10.4f} {r['mean_val']:>10.4f}"
            )
        print()

    # Artifact types
    if artifact_stats:
        print("  Artifact Types :")
        for r in artifact_stats:
            print(f"    • {r['artifact_type']:<16} {r['count']:>6,}  ({_human_bytes(r['total_bytes'])})")
        print()

    # Dataset event types
    if event_stats:
        print("  Dataset Events :")
        for r in event_stats:
            print(f"    • {r['event_type']:<18} {r['count']:>6,}")
        print()

    # Activity sparkline (last 30 days)
    if activity:
        max_runs = max(r["runs"] for r in activity)
        print("  Run Activity (last 30 days) :")
        for r in activity[:15]:
            bar = _sparkbar(r["runs"], max_runs, width=20)
            print(f"    {r['day']}  {bar}  {r['runs']:>4}")
        print()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def metrics_summary(key: str = "", experiment_name: str = "") -> None:
    if not os.path.exists(DB_PATH):
        print("ℹ️  Tracking database not found.")
        return

    sql = load_sections("metric_summary.sql")
    conn = _conn()

    if key:
        rows = conn.execute(sql["key_detail"], {"key": key, "experiment_name": experiment_name}).fetchall()
        conn.close()

        print()
        print("=" * 68)
        print(f"  Metric Detail: {key}")
        if experiment_name:
            print(f"  Experiment  : {experiment_name}")
        print("=" * 68)

        if not rows:
            print(f"  ℹ️  No data found for metric '{key}'.")
            print()
            return

        print(f"  Runs tracking this metric: {len(rows)}")
        print()
        hdr = f"    {'Run ID':<10} {'Experiment':<30} {'Min':>10} {'Max':>10} {'Mean':>10} {'Steps':>7}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for r in rows:
            print(
                f"    {r['run_id'][:8]:<10} {r['experiment'][:30]:<30}"
                f" {r['min_val']:>10.4f} {r['max_val']:>10.4f} {r['mean_val']:>10.4f} {r['steps']:>7,}"
            )
        print()
    else:
        rows = conn.execute(sql["all_keys"], {"experiment_name": experiment_name}).fetchall()
        conn.close()

        print()
        print("=" * 68)
        print("  Metric Summary")
        if experiment_name:
            print(f"  Experiment : {experiment_name}")
        print("=" * 68)

        if not rows:
            print("  ℹ️  No metrics recorded yet.")
            print()
            return

        hdr = f"    {'Key':<36} {'Runs':>5} {'Steps':>8} {'Min':>10} {'Max':>10} {'Mean':>10} {'Span':>6}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for r in rows:
            print(
                f"    {r['key'][:36]:<36} {r['runs']:>5} {r['total_steps']:>8,}"
                f" {r['min_val']:>10.4f} {r['max_val']:>10.4f} {r['mean_val']:>10.4f} {r['step_span']:>6}"
            )
        print()
