# Changelog

All notable changes to Build-Optimization are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); items graduate here from [`moon/ROADMAP.md`](ROADMAP.md) once merged.

## [Unreleased]

### Added (SA5 — Seeded Monte-Carlo projected-run planner)

- Added `middleware/src/core/mc_planner.py` (SA5): a pure, core-only, **seeded** Monte-Carlo projected-run planner for the STS2 reward advisor. It samples remaining *rarity-weighted* future card rewards (and, when gold is provided, a minimal shop buy-vs-remove sub-model) across the Act/floor horizon, **without playing combats**, and reports a **mean + confidence band** per candidate action `{Skip, offer A, offer B, offer C}`. Determinism is the contract: the same `seed`, deck and catalogue produce byte-for-byte identical bands (verified by unit tests). `slot_bonus` is neutralised in the default scorer so the projected axis doesn't inherit V1-V8's fill-the-deck bias (per Grok's scoping; SA3/SA4 own the concrete `score_build` deltas and Pareto front).
- Added `middleware/tests/test_mc_planner.py` (SA5): 9 unit tests covering determinism, Skip-vs-Take semantics, rarity-weighted sampling, the shop buy-vs-remove sub-model, and result/band bounds. All deterministic under pinned seeds.

### Added (T7 — Cross-platform Tauri bundling CI)

- Added multi-platform GitHub Actions workflow `.github/workflows/package-and-build.yml` with a matrix building on `ubuntu-22.04` (Linux `.deb`, `.AppImage`), `macos-latest` (macOS `.dmg`), and `windows-latest` (Windows `.msi`, `.exe`). Configured Linux system dependencies (`libwebkit2gtk-4.1-dev`, `libayatana-appindicator3-dev`, etc.) and artifact uploads.
- Configured Tauri bundle icon paths in `frontend/src-tauri/tauri.conf.json`.

### Added (SA6/SA7 — Slay the Spire 2 Study Reward Advisor in Tauri Studio)

- **SA6 (Tauri IPC Bridge)**: Added `frontend/src-tauri/src/commands/advisor.rs` with `run_sts2_advisor` command, interfacing with `advisor_cli.py` via JSON IPC, with full deterministic analytical fallback for development and testing.
- **SA7 (Advisor Page & Visualization)**: Added `frontend/src/pages/Advisor.tsx` study screen in Tauri Studio:
  - 3-card reward offer inputs with card suggestions and upgrade indicators.
  - Deck manager with starter preset (5 Strike, 4 Defend, 1 Bash), Strength archetype, Block archetype, and blank custom deck options.
  - Run context controls (Act, Floor, HP %, Gold, Relics, Potions).
  - Preference & strategy weights (Balanced, Tempo Survival, Archetype Scaling, Anti-Dilution) with configurable Monte Carlo rollouts and RNG seed.
  - Interactive recommendation banner, ranked choice breakdown table, Pareto frontier non-dominated markers, and synergy delta explanations.
  - ECharts metric comparison bar chart and Monte Carlo confidence interval error-band visualization.
- **Navigation & Types**: Added `/advisor` route to `frontend/src/App.tsx`, sidebar navigation link in `frontend/src/components/Sidebar.tsx`, and TypeScript interfaces in `frontend/src/lib/types.ts` and `frontend/src/lib/tauriApi.ts`.
- **Tests**: Added `frontend/src/lib/__tests__/advisor.test.ts` validating request construction and response Pareto front processing. All 9 frontend tests pass cleanly.

### Added (Slay the Spire 2 Vertical Slice — V1–V8)

- **V1 (Design Doc)**: Added `docs/deck-problem-mapping.md` documenting the domain mapping between cards, decks, energy/gold costs, card type slots (`ATTACK`, `SKILL`, `POWER`), and 0-1 knapsack subset selection. Linked in `docs/mkdocs.yml`.
- **V2 (Deck Problem)**: Added `middleware/src/core/deck_problem.py` (`DeckProblem`) and `middleware/src/core/deck.py` (`Deck`) for subset selection, with `score_fast`, `score_full`, `to_deck`, and `to_result_json` producing standard result JSON payloads. Added card type slots (`ATTACK`, `SKILL`, `POWER`) to `Slot` enum and `FileSource` taxonomy map.
- **V3 (Deck Pipeline & Knapsack Solver)**: Added `middleware/src/pipeline/decks/optimizer.py` and `middleware/src/pipeline/decks/__init__.py` registering `knapsack` (calling C++ `solve_knapsack` via `core.native_backend`) and `greedy` solvers for deck optimization.
- **V4 (Ironclad Card Dataset)**: Authored `middleware/src/data/sample/slay_the_spire_2_ironclad.json` with 41 cards and 8 archetype synergy rules (Strength Scaling, Multi-Hit Mastery, Scaling Powerhouse, Block Conversion).
- **V5 (Config Profiles)**: Added `middleware/configs/game/slay_the_spire_2.yaml` (stat weights and synergy rules) and `middleware/configs/optimization/slay_the_spire_2.yaml` (deck size cap = 18). Added policy configs `policy_deck_knapsack.yaml` and `policy_deck_greedy.yaml`.
- **V6 (Hydra Routing)**: Extended `main.py` to route `problem_type: "deck"` games through `_run_deck_game` and `run_deck_optimization`, writing results to `outputs/` and logging runs to `tracking.db`. Safely sanitized non-finite float budgets (`_json_safe_float`) for strict `serde_json` compatibility.
- **V7 (Tests)**: Added `middleware/tests/test_deck_problem.py` with hand-verified unit tests, brute-force search cross-checks, and pipeline integration tests. All 29 unit tests pass cleanly.
- **V8 (Frontend & Build Verification)**: Verified that Tauri Studio frontend (`frontend/`) and TypeScript build (`npm run build`, `npm test`) compile cleanly without code changes.

### Added (B5 — backend solver wired into a selectable Hydra policy)

- Added `middleware/src/core/native_backend.py`: lazily locates and imports the compiled `build_optimizer_backend` extension from `backend/` (built by a separate Pixi toolchain, not on `sys.path` by default), with a clear `ImportError` and build instructions if it isn't built yet.
- Registered a `bnb` solver in `pipeline/games/states/solving.py`: converts `BuildProblem` into the C++ solver's `MckpOption` list (class = slot, weight = cost, value = the same per-item `score_fast` contribution every other solver optimizes internally), calls `solve_mckp_branch_and_bound`, converts the result back into a `slot_to_item` array. Added `middleware/configs/policy/policy_bnb.yaml` and `bnb` to `main.py`'s `_PIPELINE_SOLVERS`, so `policy=policy_bnb` is selectable end-to-end from the CLI.
- Verified with real runs, not just unit tests: on the RPG sample data at the default budget it reaches the *same* optimum SA already found (1037.1); at a tighter budget (1200) it finds a **materially better** solution than greedy on the shared fast-score objective (79.4 vs 28.7) — the expected signature of an exact solver beating a heuristic once the problem is more constrained.
- Extended `.github/workflows/ci.yml` so `test-python` actually exercises this: `build-backend` now uploads the compiled `.so` as an artifact, `test-python` downloads it and pins `uv sync --python 3.11` to match Pixi's build ABI. `middleware/tests/test_native_backend.py`'s backend-dependent tests skip gracefully (not fail) when the artifact isn't present.
- Corrected two stale README sections found while updating them: the "Available Solvers" table listed 8 solvers (`random`, `ils`, `lahc`, `rrt`, `gls`, `rts`, `oba`, `alns`) that were never actually registered in `pipeline/games/states/solving.py` (only `greedy`/`sa`/`ga` — the others are just aliased to one of those three in `main.py`'s `_SOLVER_ALIAS`); and the "Native Solver Bindings" capability row claimed backend calls went through `middleware/src/policies`, which isn't where this wiring lives.

### Added (B2 — branch-and-bound MCKP solver)

- Added `backend/include/build_optimizer/mckp.hpp` / `backend/src/mckp.cpp`: an exact solver for the **Multiple-Choice Knapsack Problem** — classes of mutually-exclusive options (equipment slots, directly matching `Build`'s one-item-per-slot structure) with at most one selected per class, subject to a shared weight/cost capacity. Uses depth-first branch-and-bound with an admissible fractional-relaxation upper bound (Dantzig's classic 0-1 knapsack bound, generalized to MCKP) for pruning — genuinely different from, and complementary to, the existing DP-based plain 0-1 `solve_knapsack`.
- Exposed via pybind11 (`MckpOption`, `MckpResult`, `solve_mckp_branch_and_bound`) and verified callable from Python, not just compiled.
- Wired `backend/CMakeLists.txt` to build a Catch2 test executable (`backend/tests/test_knapsack.cpp`, `backend/tests/test_mckp.cpp`) when the `dev` Pixi environment is active, registered with CTest via `catch_discover_tests` — `pixi run -e dev ctest --test-dir build` (10 tests) now actually runs, which it did not before (the `pixi.toml` `test` task existed but no tests were ever registered). Includes a brute-force cross-check test for the MCKP solver on a 4-class/12-option instance.

### Added (T6 — Tauri commands for the tracking database)

- Added `frontend/src-tauri/src/commands/tracking.rs`: six `sqlx`-backed commands (`list_experiments`, `list_tracked_runs`, `get_run_params`, `get_run_latest_metrics`, `get_run_metric_history`, `get_run_artifacts`) reading `assets/tracking/tracking.db` read-only, mirroring `TrackingStore`'s Python query surface. All degrade to an empty result (not an error) when the database doesn't exist yet.
- Verified against a real database (not just compiled): a `#[tokio::test]` integration test runs every query against the `tracking.db` produced by an actual `main.py` run and checks the returned params/metrics/artifact match.
- Added a `TrackedRunsPanel` component and wired it into Build Explorer (collapsible "📊 Tracked Runs" section) — browse experiments/runs from the database and load a run's result via its logged artifact path, reusing the existing `BuildDetail` view.

### Added (B11 — tracking database, wired end-to-end)

- Wired `middleware/src/pipeline/games/optimizer.py::run_optimization`/`run_batch` to persist every solve: a result JSON under `outputs/<experiment>/<solver>_<timestamp>_result.json` (consumed by the Tauri Studio's existing file-based commands) *and* a full record — experiment, run, params, metrics, artifact link — in the `tracking.db` SQLite database (schema was already present, inherited from WSmart-Route, but was never actually exercised for Build-Optimization). New `experiment_name`/`persist` parameters (defaults: game name, `True`).
- Added `middleware/tests/test_tracking.py`: round-trip store tests plus an integration test that runs the full CLI pipeline and asserts both the output file and the tracking DB are written correctly.

### Fixed (making the CLI actually runnable — none of this worked before)

- `middleware/src/constants/paths.py`'s `ROOT_DIR` resolution hardcoded `"WSmart-Route"`/`"WSmartPlus-Route"` as the only recognized clone directory names — importing `constants` in Build-Optimization crashed immediately with an uncaught `ValueError`. Added `"Build-Optimization"` plus a `pyproject.toml`-marker fallback so it works under any clone name.
- Root `pyproject.toml` had no `[tool.setuptools]` package config; `uv sync` failed outright once the repo grew past a couple of top-level directories (`setuptools` can't auto-discover a package layout among `git/`, `moon/`, `backend/`, `frontend/`, `extension/`, `archive/`, `research/`, `infra/global/docker/`, `desktop/`, ...). Added `packages = []` — this pyproject only manages dependencies for `main.py`, nothing here needs to be built as an installable package.
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
