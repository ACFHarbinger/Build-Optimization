# Changelog

All notable changes to Build-Optimization are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); items graduate here from [`moon/ROADMAP.md`](ROADMAP.md) once merged.

## [Unreleased]

### Added (T6 — Tauri commands for the tracking database)

- Added `frontend/src-tauri/src/commands/tracking.rs`: six `sqlx`-backed commands (`list_experiments`, `list_tracked_runs`, `get_run_params`, `get_run_latest_metrics`, `get_run_metric_history`, `get_run_artifacts`) reading `assets/tracking/tracking.db` read-only, mirroring `TrackingStore`'s Python query surface. All degrade to an empty result (not an error) when the database doesn't exist yet.
- Verified against a real database (not just compiled): a `#[tokio::test]` integration test runs every query against the `tracking.db` produced by an actual `main.py` run and checks the returned params/metrics/artifact match.
- Added a `TrackedRunsPanel` component and wired it into Build Explorer (collapsible "📊 Tracked Runs" section) — browse experiments/runs from the database and load a run's result via its logged artifact path, reusing the existing `BuildDetail` view.

### Added (B11 — tracking database, wired end-to-end)

- Wired `middleware/src/pipeline/games/optimizer.py::run_optimization`/`run_batch` to persist every solve: a result JSON under `outputs/<experiment>/<solver>_<timestamp>_result.json` (consumed by the Tauri Studio's existing file-based commands) *and* a full record — experiment, run, params, metrics, artifact link — in the `tracking.db` SQLite database (schema was already present, inherited from WSmart-Route, but was never actually exercised for Build-Optimization). New `experiment_name`/`persist` parameters (defaults: game name, `True`).
- Added `middleware/tests/test_tracking.py`: round-trip store tests plus an integration test that runs the full CLI pipeline and asserts both the output file and the tracking DB are written correctly.

### Fixed (making the CLI actually runnable — none of this worked before)

- `middleware/src/constants/paths.py`'s `ROOT_DIR` resolution hardcoded `"WSmart-Route"`/`"WSmartPlus-Route"` as the only recognized clone directory names — importing `constants` in Build-Optimization crashed immediately with an uncaught `ValueError`. Added `"Build-Optimization"` plus a `pyproject.toml`-marker fallback so it works under any clone name.
- Root `pyproject.toml` had no `[tool.setuptools]` package config; `uv sync` failed outright once the repo grew past a couple of top-level directories (`setuptools` can't auto-discover a package layout among `git/`, `moon/`, `backend/`, `frontend/`, `extension/`, `archive/`, `research/`, `docker/`, `desktop/`, ...). Added `packages = []` — this pyproject only manages dependencies for `main.py`, nothing here needs to be built as an installable package.
- `main.py` didn't add `middleware/src` to `sys.path` and pointed Hydra at a `configs/` directory that doesn't exist at the repo root (configs live at `middleware/configs/`) — `python main.py` could not run at all. Fixed both.
- `middleware/configs/game/rpg.yaml` and `moba.yaml` pointed at `src/data/sample_games/*.json`, a directory that has never existed (the real data lives at `middleware/src/data/sample/`); `moba.json` itself didn't exist. Fixed both config paths and added a sample `moba.json` dataset. Also made `FileSource`'s path resolution robust to being invoked from any working directory (falls back to resolving relative to `middleware/` under `ROOT_DIR`).
- `middleware/src/tracking`'s package `__init__.py` (and `integrations/__init__.py`, `profiling/__init__.py`) eagerly imported PyTorch/PyTorch-Lightning/`lightning.fabric`-dependent submodules (`hooks`, `integrations.data`, `integrations.lightning`, `integrations.zenml_bridge`) at import time, so `import tracking` failed in any environment without the full ML stack installed — including Build-Optimization's lean solver-only setup. Made those four lazy (PEP 562 module `__getattr__`) so the tracking core works standalone; they still work normally once torch/lightning are installed.
- Added missing `__init__.py` to 4 `middleware/src` subpackages (`policies/helpers/hpo`, `policies/route_improvement/common`, `pipeline/rl/meta/multi_objective`, `policies/route_construction/meta_heuristics/simulated_annealing`) — the last one has a relative import that broke sphinx-autoapi's static resolution without it (see the docs-tooling commit).
- Removed `middleware/tests/test_solvers.py`, which imported a `solvers.*` package that has not existed since the WSmart-Route policy-tree merge (the real greedy/SA/GA implementations live as registered functions in `pipeline/games/states/solving.py`); it could never pass and blocked test collection entirely.

### Added

- Rewrote `README.md` with tech-stack badges, a full architecture diagram, and setup/run/test instructions for the C++ backend, Python middleware, Tauri + TypeScript frontend, and browser extension/engine integrations.
- Added `git/codecov.yaml` and `git/CONTRIBUTING.md`.
- Added `moon/ROADMAP.md` and `moon/CHANGELOG.md`.
- Added `backend/CMakeLists.txt` and `backend/pixi.toml` scaffolding the C++ solver module.
- Added `middleware/pyproject.toml` for the Python middleware package.
- Added `extension/package.json` (Manifest V3 browser extension for wiki data scraping) and `frontend/package.json` (Tauri + TypeScript frontend).
- Added root `package.json` (NPM workspace over `frontend/` and `extension/`) and root `Cargo.toml` (Cargo workspace over `frontend/src-tauri`).
- Mirrored repository infrastructure conventions from Image-Toolkit and WSmart-Route: Docker build definitions, CI/CD workflows, `docs/` references, the `.agent/` AI-assistant guide, `env/` templates, and `desktop/` platform scripts.
- Implemented the Tauri Studio's data-access layer: `frontend/src-tauri/src/commands/` (`list_solver_results`, `read_solver_result`, `list_item_files`, `read_items_json`, `list_training_runs`, `read_training_log`), reading `outputs/`/`data/` directly.
- Implemented the Build Explorer, Solver Comparison, Training Monitor, and Item Database pages in `frontend/src/pages/`, plus shared `KpiRow`/`Sidebar` components, an ECharts-based radar/bar/line/boxplot/scatter chart set, `HashRouter`-based navigation, and a light/dark-aware CSS port of the original dashboard's stylesheet.
- Added a unified MkDocs Material documentation portal (`docs/mkdocs.yml`) that embeds `README.md`/`git/CONTRIBUTING.md`/`moon/ROADMAP.md`/`moon/CHANGELOG.md` via `pymdownx.snippets`, plus five per-language API generators — sphinx-autoapi (Python), Doxygen (C++), TypeDoc (Tauri frontend + browser extension), and rustdoc (Tauri Rust shell) — orchestrated by `docs/build_docs.sh` and `just docs`. Wired into `.github/workflows/docs.yml` (per-generator jobs + GitHub Pages deploy).

### Changed

- Renamed the Tauri desktop module from `app/` to `frontend/`, reserving `app/` for a future Kotlin Android module.
- Reworked `extension/` from a VS Code extension scaffold into a Manifest V3 browser extension (Chrome/Firefox/Edge) that scrapes item/stat data from game wikis (Fandom, wiki.gg) for `middleware/src/pipeline/file_source.py`.
- Migrated the Streamlit Control Tower dashboard's functionality (Build Explorer, Solver Comparison, Training Monitor, Item Database) from `middleware/ui` into `frontend/` as native React pages; the `streamlit` dependency is dropped from `middleware/pyproject.toml`. The original Python implementation is preserved at [`archive/middleware/`](../archive/middleware/) for reference.
- `just dashboard` is renamed to `just studio`, launching the Tauri Studio in dev mode.
- `extension/tsconfig.json`'s `moduleResolution` changed from the deprecated `node` to `bundler`, matching `frontend/tsconfig.json`.

### Fixed

- Added missing `__init__.py` files in four `middleware/src` subpackages (`policies/helpers/hpo`, `policies/route_improvement/common`, `pipeline/rl/meta/multi_objective`, `policies/route_construction/meta_heuristics/simulated_annealing`) — the last one had a relative import that sphinx-autoapi couldn't statically resolve without it.
- Added `frontend/src/vite-env.d.ts` (missing `vite/client` type reference, needed for `import "./index.css"` side-effect imports to type-check under TypeDoc's stricter checker).

## [0.1.0] - Initial solver suite

### Added

- Hydra-driven `main.py` entry point with `greedy` / `sa` / `ga` pipeline solvers.
- Core domain model (`Item`, `Build`, `SynergyEngine`, scoring) under `middleware/src/core`.
- 11 native solvers (greedy through ALNS) plus 28 metaheuristic policies carried over from WSmart-Route.
- File/API/scraper data pipeline skeleton and sample RPG item dataset.
- Streamlit control-tower dashboard (`middleware/ui`).
- `tools/*/justfile` command-runner modules and the root `justfile` dispatcher.
