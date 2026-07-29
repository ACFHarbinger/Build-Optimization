use serde::Serialize;
use serde_json::Value;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::path::PathBuf;

use super::repo_root;

/// Path to the SQLite database written by `middleware/src/tracking`
/// (`Tracker(tracking_uri=...)`, defaulting to `<repo_root>/assets/tracking`).
fn tracking_db_path() -> PathBuf {
    repo_root().join("assets").join("tracking").join("tracking.db")
}

async fn connect() -> Option<SqlitePool> {
    let path = tracking_db_path();
    if !path.exists() {
        return None;
    }
    let url = format!("sqlite:{}?mode=ro", path.display());
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&url)
        .await
        .ok()
}

#[derive(Serialize)]
pub struct ExperimentRow {
    id: i64,
    name: String,
    created_at: String,
    description: String,
}

/// Lists experiments (an "experiment" corresponds to a game, e.g. "Fantasy RPG").
/// Returns an empty list, not an error, if the tracking database doesn't exist yet
/// (no optimization run has happened).
#[tauri::command]
pub async fn list_experiments() -> Result<Vec<ExperimentRow>, String> {
    let Some(pool) = connect().await else {
        return Ok(Vec::new());
    };
    let rows = sqlx::query("SELECT id, name, created_at, description FROM experiments ORDER BY created_at DESC")
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|r| ExperimentRow {
            id: r.get("id"),
            name: r.get("name"),
            created_at: r.get("created_at"),
            description: r.get("description"),
        })
        .collect())
}

#[derive(Serialize)]
pub struct RunRow {
    id: String,
    experiment_name: String,
    name: Option<String>,
    status: String,
    run_type: String,
    start_time: String,
    end_time: Option<String>,
}

/// Lists runs, optionally scoped to one experiment (game) name, newest first.
#[tauri::command]
pub async fn list_tracked_runs(experiment_name: Option<String>) -> Result<Vec<RunRow>, String> {
    let Some(pool) = connect().await else {
        return Ok(Vec::new());
    };

    let base = "SELECT r.id, e.name AS experiment_name, r.name, r.status, r.run_type, r.start_time, r.end_time \
                FROM runs r JOIN experiments e ON r.experiment_id = e.id";
    let rows = if let Some(name) = experiment_name {
        sqlx::query(&format!("{base} WHERE e.name = ? ORDER BY r.start_time DESC"))
            .bind(name)
            .fetch_all(&pool)
            .await
    } else {
        sqlx::query(&format!("{base} ORDER BY r.start_time DESC")).fetch_all(&pool).await
    }
    .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|r| RunRow {
            id: r.get("id"),
            experiment_name: r.get("experiment_name"),
            name: r.get("name"),
            status: r.get("status"),
            run_type: r.get("run_type"),
            start_time: r.get("start_time"),
            end_time: r.get("end_time"),
        })
        .collect())
}

/// Returns all logged parameters for a run, JSON-decoded (params are stored
/// as JSON-encoded strings, matching `TrackingStore.get_params` in Python).
#[tauri::command]
pub async fn get_run_params(run_id: String) -> Result<serde_json::Map<String, Value>, String> {
    let Some(pool) = connect().await else {
        return Ok(serde_json::Map::new());
    };
    let rows = sqlx::query("SELECT key, value FROM params WHERE run_id = ?")
        .bind(run_id)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?;

    let mut out = serde_json::Map::new();
    for row in rows {
        let key: String = row.get("key");
        let raw: String = row.get("value");
        let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
        out.insert(key, value);
    }
    Ok(out)
}

/// Returns the latest value for every metric key logged on a run
/// (matches `TrackingStore.get_latest_metrics` in Python).
#[tauri::command]
pub async fn get_run_latest_metrics(run_id: String) -> Result<serde_json::Map<String, Value>, String> {
    let Some(pool) = connect().await else {
        return Ok(serde_json::Map::new());
    };
    let rows = sqlx::query(
        "SELECT key, value FROM metrics \
         WHERE id IN (SELECT MAX(id) FROM metrics WHERE run_id = ? GROUP BY key)",
    )
    .bind(run_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut out = serde_json::Map::new();
    for row in rows {
        let key: String = row.get("key");
        let value: f64 = row.get("value");
        out.insert(key, Value::from(value));
    }
    Ok(out)
}

#[derive(Serialize)]
pub struct MetricPoint {
    value: f64,
    step: i64,
    timestamp: String,
}

/// Returns the full step-indexed history for one metric key on a run —
/// e.g. for future training-curve visualizations (see moon/ROADMAP.md T4).
#[tauri::command]
pub async fn get_run_metric_history(run_id: String, key: String) -> Result<Vec<MetricPoint>, String> {
    let Some(pool) = connect().await else {
        return Ok(Vec::new());
    };
    let rows = sqlx::query("SELECT value, step, timestamp FROM metrics WHERE run_id = ? AND key = ? ORDER BY step")
        .bind(run_id)
        .bind(key)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|r| MetricPoint {
            value: r.get("value"),
            step: r.get("step"),
            timestamp: r.get("timestamp"),
        })
        .collect())
}

#[derive(Serialize)]
pub struct ArtifactRow {
    name: String,
    path: String,
    artifact_type: String,
    created_at: String,
}

/// Returns artifacts (e.g. result JSON files) registered against a run.
/// `path` can be passed directly to `read_solver_result`/`read_items_json`.
#[tauri::command]
pub async fn get_run_artifacts(run_id: String) -> Result<Vec<ArtifactRow>, String> {
    let Some(pool) = connect().await else {
        return Ok(Vec::new());
    };
    let rows = sqlx::query("SELECT name, path, artifact_type, created_at FROM artifacts WHERE run_id = ? ORDER BY created_at")
        .bind(run_id)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|r| ArtifactRow {
            name: r.get("name"),
            path: r.get("path"),
            artifact_type: r.get("artifact_type"),
            created_at: r.get("created_at"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises every query against the real tracking.db written by
    /// `uv run python main.py policy=policy_sa game=rpg`. Skips (not fails)
    /// if that hasn't been run locally / in CI — this is a real-schema
    /// integration check, not a hermetic unit test.
    #[tokio::test]
    async fn queries_match_the_python_schema() {
        if !tracking_db_path().exists() {
            eprintln!("skipping: no tracking.db at {:?} (run `python main.py policy=policy_sa game=rpg` first)", tracking_db_path());
            return;
        }

        let experiments = list_experiments().await.expect("list_experiments");
        assert!(!experiments.is_empty(), "expected at least one experiment");

        let runs = list_tracked_runs(Some(experiments[0].name.clone())).await.expect("list_tracked_runs");
        assert!(!runs.is_empty(), "expected at least one run");
        let run_id = runs[0].id.clone();

        let params = get_run_params(run_id.clone()).await.expect("get_run_params");
        assert!(params.contains_key("solver"), "expected a 'solver' param, got {params:?}");

        let metrics = get_run_latest_metrics(run_id.clone()).await.expect("get_run_latest_metrics");
        assert!(metrics.contains_key("score"), "expected a 'score' metric, got {metrics:?}");

        let artifacts = get_run_artifacts(run_id).await.expect("get_run_artifacts");
        assert_eq!(artifacts.len(), 1, "expected exactly one result-JSON artifact");
        assert!(std::path::Path::new(&artifacts[0].path).exists(), "artifact path should exist on disk");
    }
}
