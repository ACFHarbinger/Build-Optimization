# Changelog

All notable changes to Build-Optimization are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); items graduate here from [`.moon/ROADMAP.md`](ROADMAP.md) once merged.

## [Unreleased]

### Added

- Rewrote `README.md` with tech-stack badges, a full architecture diagram, and setup/run/test instructions for the C++ backend, Python middleware, Tauri + TypeScript frontend, and IDE/engine integrations.
- Added `.github/codecov.yaml` and `.github/CONTRIBUTING.md`.
- Added `.moon/ROADMAP.md` and `.moon/CHANGELOG.md`.
- Added `backend/CMakeLists.txt` and `backend/pixi.toml` scaffolding the C++ solver module.
- Added `middleware/pyproject.toml` for the Python middleware package.
- Added `extension/package.json` (VS Code extension) and `app/package.json` (Tauri + TypeScript frontend).
- Added root `package.json` (NPM workspace over `app/` and `extension/`) and root `Cargo.toml` (Cargo workspace over `app/src-tauri`).
- Mirrored repository infrastructure conventions from Image-Toolkit and WSmart-Route: Docker build definitions, CI/CD workflows, `docs/` references, the `.agent/` AI-assistant guide, `env/` templates, and `desktop/` platform scripts.

## [0.1.0] - Initial solver suite

### Added

- Hydra-driven `main.py` entry point with `greedy` / `sa` / `ga` pipeline solvers.
- Core domain model (`Item`, `Build`, `SynergyEngine`, scoring) under `middleware/src/core`.
- 11 native solvers (greedy through ALNS) plus 28 metaheuristic policies carried over from WSmart-Route.
- File/API/scraper data pipeline skeleton and sample RPG item dataset.
- Streamlit control-tower dashboard (`middleware/ui`).
- `tools/*/justfile` command-runner modules and the root `justfile` dispatcher.
