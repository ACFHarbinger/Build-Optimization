use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

use super::repo_root;

const RUN_MARKERS: [&str; 3] = ["metrics.csv", "training_log.jsonl", "training_log.csv"];

fn find_run_dirs(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut has_marker = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_run_dirs(&path, out);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if RUN_MARKERS.contains(&name) {
                has_marker = true;
            }
        }
    }
    if has_marker {
        out.push(dir.to_string_lossy().to_string());
    }
}

/// Finds training run directories (containing `metrics.csv` or `training_log.jsonl`)
/// under `<repo_root>/outputs/`.
#[tauri::command]
pub fn list_training_runs() -> Vec<String> {
    let dir = repo_root().join("outputs");
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    find_run_dirs(&dir, &mut runs);
    runs.sort();
    runs
}

fn parse_csv(content: &str) -> Vec<Map<String, Value>> {
    let mut lines = content.lines();
    let Some(header_line) = lines.next() else {
        return Vec::new();
    };
    let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();

    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut record = Map::new();
            for (header, field) in headers.iter().zip(line.split(',')) {
                let field = field.trim();
                let value = field
                    .parse::<f64>()
                    .map(|n| Value::from(n))
                    .unwrap_or_else(|_| Value::String(field.to_string()));
                record.insert(header.to_string(), value);
            }
            record
        })
        .collect()
}

/// Parses a training run directory into a list of record objects.
/// Supports `metrics.csv`, `training_log.jsonl` (one JSON object per line),
/// and `training_log.csv`, checked in that order.
#[tauri::command]
pub fn read_training_log(run_dir: String) -> Result<Vec<Map<String, Value>>, String> {
    let dir = Path::new(&run_dir);

    let csv_path = dir.join("metrics.csv");
    if csv_path.exists() {
        let content = fs::read_to_string(&csv_path).map_err(|e| e.to_string())?;
        return Ok(parse_csv(&content));
    }

    let jsonl_path = dir.join("training_log.jsonl");
    if jsonl_path.exists() {
        let content = fs::read_to_string(&jsonl_path).map_err(|e| e.to_string())?;
        let records = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str::<Map<String, Value>>(line).ok())
            .collect();
        return Ok(records);
    }

    let csv2_path = dir.join("training_log.csv");
    if csv2_path.exists() {
        let content = fs::read_to_string(&csv2_path).map_err(|e| e.to_string())?;
        return Ok(parse_csv(&content));
    }

    Ok(Vec::new())
}
