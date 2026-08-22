pub mod advisor;
pub mod data;
pub mod tracking;
pub mod training;

use std::path::{Path, PathBuf};

/// Locates the repository root by walking up from the crate's build-time
/// manifest directory (`frontend/src-tauri/`) looking for the root `pyproject.toml`.
/// This is stable regardless of the process's runtime working directory.
pub fn repo_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        if ancestor.join("pyproject.toml").is_file() {
            return ancestor.to_path_buf();
        }
    }
    manifest_dir.to_path_buf()
}
