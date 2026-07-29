use std::fs;
use std::path::Path;

use serde_json::Value;

use super::repo_root;

fn walk_files(dir: &Path, matches: impl Fn(&str) -> bool + Copy) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk_files(&path, matches));
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if matches(name) {
                out.push(path.to_string_lossy().to_string());
            }
        }
    }
    out
}

/// Finds solver result JSON files under `<repo_root>/outputs/`.
#[tauri::command]
pub fn list_solver_results() -> Vec<String> {
    let dir = repo_root().join("outputs");
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut results = walk_files(&dir, |name| {
        let lower = name.to_lowercase();
        lower.ends_with(".json") && (lower.contains("result") || lower.contains("solution"))
    });
    results.sort();
    results
}

/// Reads a single solver result JSON file.
#[tauri::command]
pub fn read_solver_result(path: String) -> Result<Value, String> {
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&content).map_err(|e| e.to_string())
}

/// Finds item data files (JSON/CSV containing "item" in the name) under `<repo_root>/data/`
/// and `middleware/src/data/sample/`.
#[tauri::command]
pub fn list_item_files() -> Vec<String> {
    let root = repo_root();
    let mut dirs = vec![root.join("data")];
    let sample_dir = root.join("middleware/src/data/sample");
    if sample_dir.is_dir() {
        dirs.push(sample_dir);
    }

    let mut results = Vec::new();
    for dir in dirs {
        if dir.is_dir() {
            results.extend(walk_files(&dir, |name| {
                let lower = name.to_lowercase();
                (lower.ends_with(".json") || lower.ends_with(".csv")) && lower.contains("item")
            }));
            // Sample game datasets (rpg.json, moba.json, darktide.json) don't have
            // "item" in the filename but hold the same `{"items": [...]}` schema.
            results.extend(walk_files(&dir, |name| {
                let lower = name.to_lowercase();
                lower.ends_with(".json") && !lower.contains("item")
            }));
        }
    }
    results.sort();
    results.dedup();
    results
}

/// Loads item data from a JSON file. Accepts either a top-level array of
/// items or an object with an `items` key (matches `FileSource`'s schema).
#[tauri::command]
pub fn read_items_json(path: String) -> Result<Vec<Value>, String> {
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let data: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

    match data {
        Value::Array(items) => Ok(items),
        Value::Object(mut obj) => match obj.remove("items") {
            Some(Value::Array(items)) => Ok(items),
            _ => Ok(Vec::new()),
        },
        _ => Ok(Vec::new()),
    }
}
