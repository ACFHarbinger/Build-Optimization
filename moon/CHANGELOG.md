# Changelog

All notable changes to Build-Optimization are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); items graduate here from [`moon/ROADMAP.md`](ROADMAP.md) once merged.

## [Unreleased]

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
