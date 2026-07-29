<div align="center">

# Build-Optimization

**A cross-platform optimization framework for videogame character builds — combining exact and metaheuristic knapsack-routing solvers with a live desktop studio.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)

</br>

[![pybind11](https://img.shields.io/badge/pybind11-Bridge-3776AB?logo=cplusplus&logoColor=white)](https://pybind11.readthedocs.io/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Pixi](https://img.shields.io/badge/Pixi-Env_Manager-F9A03C?logo=conda-forge&logoColor=white)](https://pixi.sh/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-blue)](https://hydra.cc/)
[![Cargo](https://img.shields.io/badge/Cargo-E57300?logo=rust&logoColor=white)](https://doc.rust-lang.org/cargo/)
[![NPM](https://img.shields.io/badge/NPM-Workspaces-CB3837?logo=npm&logoColor=white)](https://docs.npmjs.com/cli/v10/using-npm/workspaces)
[![Just](https://img.shields.io/badge/Just-Task_Runner-black)](https://github.com/casey/just)

</br>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Vitest](https://img.shields.io/badge/vitest-testing-6E9F18?logo=vitest&logoColor=white)](https://vitest.dev/)
[![Coverage](https://img.shields.io/badge/coverage-60%25-green.svg)](https://codecov.io/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot&logoColor=white)](https://dependabot.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p>
  <a href="#overview"><strong>Overview</strong></a> |
  <a href="#tech-stack--capabilities"><strong>Tech Stack</strong></a> |
  <a href="#architecture"><strong>Architecture</strong></a> |
  <a href="#installation--setup"><strong>Setup</strong></a> |
  <a href="#running-the-project"><strong>Running</strong></a> |
  <a href="#testing"><strong>Testing</strong></a> |
  <a href="#documentation"><strong>Documentation</strong></a> |
  <a href="#contributing"><strong>Contributing</strong></a>
</p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack & Capabilities](#tech-stack--capabilities)
  - [C++ Backend](#c-backend)
  - [Python Middleware](#python-middleware)
  - [Tauri + TypeScript Frontend](#tauri--typescript-frontend)
  - [IDE & Engine Integrations](#ide--engine-integrations)
- [Architecture](#architecture)
- [Domain Mapping](#domain-mapping)
- [Available Solvers](#available-solvers)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Build-Optimization** applies operations-research techniques from combinatorial routing (its sibling project, [WSmart-Route](https://github.com/ACFHarbinger/WSmart-Route)) to the domain of videogame character "theorycrafting" — picking the item, skill, and stat combination that maximizes a character's effectiveness under a strict resource budget.

The underlying research (see [`research/`](research/)) frames build optimization as a **lexicographic, multi-objective knapsack problem**, and in its most general form, a **knapsack-routing hybrid**: item selection ("what to equip") is a 0-1 / multiple-choice knapsack with non-linear synergy terms (set bonuses, multiplicative stat interactions), while build *paths* — skill trees, talent orders, itemization sequences — introduce routing-like precedence and sequencing constraints analogous to the Thief Orienteering Problem. Exact methods (MILP, Benders decomposition), evolutionary metaheuristics (GA, ALNS, ILS), and learned policies (PPO/MCTS for real-time MOBA drafting, GFlowNets for amortized inference) are all first-class citizens.

The project is organized as a polyglot monorepo:

| Layer | Role |
| --- | --- |
| **C++ Backend** | Performance-critical exact and combinatorial solvers, exposed to Python via `pybind11`. |
| **Python Middleware** | Hydra-driven orchestration: the existing solver suite, data pipeline, experiment tracking, and Streamlit dashboard. |
| **Tauri + TypeScript Frontend** | Cross-platform desktop "Studio" for exploring builds, comparing solvers, and monitoring training runs. |
| **IDE / Engine Integrations** | A VS Code extension for authoring build configs, and a planned Unreal Engine plugin for in-editor optimization. |

## Tech Stack & Capabilities

### C++ Backend

[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/) [![pybind11](https://img.shields.io/badge/pybind11-Bridge-3776AB?logo=cplusplus&logoColor=white)](https://pybind11.readthedocs.io/) [![CMake](https://img.shields.io/badge/CMake-Build-064F8C?logo=cmake&logoColor=white)](https://cmake.org/) [![Pixi](https://img.shields.io/badge/Pixi-Env_Manager-F9A03C?logo=conda-forge&logoColor=white)](https://pixi.sh/)

Lives in [`backend/`](backend/). Ships as a `pybind11` extension module consumed directly by the Python middleware.

| Capability | Description |
| --- | --- |
| **Exact Knapsack Solvers** | Branch-and-bound and dynamic-programming solvers for 0-1 / multiple-choice / quadratic knapsack formulations. |
| **Lexicographic Optimization** | Sequential goal-programming solve order for ranked objectives (DPS → EHP → cost, etc.). |
| **Synergy Linearization** | McCormick envelope / Fortet-inequality linearization of multiplicative set-bonus terms. |
| **Native Solver Bindings** | `pybind11`-exposed entry points callable from `middleware/src/policies` alongside the pure-Python solvers. |

### Python Middleware

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/) [![Hydra](https://img.shields.io/badge/Hydra-1.3-blue)](https://hydra.cc/) [![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv) [![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

Lives in [`middleware/`](middleware/). Owns its own [`pyproject.toml`](middleware/pyproject.toml).

| Capability | Description |
| --- | --- |
| **Solver Suite** | 11 native solvers (greedy → ALNS, see [Available Solvers](#available-solvers)) plus 28 metaheuristic policies inherited from WSmart-Route. |
| **Hydra Config System** | Composable `game` / `optimization` / `pipeline` / `policy` configs under [`middleware/configs/`](middleware/configs/). |
| **Data Pipeline** | `FileSource`, `GameAPISource`, and `WebScraperSource` ingestion backends with shared normalization transforms. |
| **Experiment Tracking** | SQLite-backed run/metric tracking (`middleware/src/tracking`), inspectable via `just database::db-*` recipes. |
| **Control Tower Dashboard** | Streamlit UI (`middleware/ui`) for build exploration, solver comparison, training monitoring, and item browsing. |
| **Static Validation** | Import-cycle, interface-compliance, and type-coverage checks (`middleware/validation`). |

### Tauri + TypeScript Frontend

[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/) [![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/) [![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/) [![Vite](https://img.shields.io/badge/Vite-Build-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)

Lives in [`app/`](app/) (Rust shell in `app/src-tauri/`). Ships its own [`package.json`](app/package.json).

| Capability | Description |
| --- | --- |
| **Build Explorer** | Interactive build inspection with synergy and effectiveness breakdowns (native port of `middleware/ui/pages/build_explorer.py`). |
| **Solver Comparison** | Side-by-side solver benchmarking, mirroring `middleware/ui/pages/solver_comparison.py`. |
| **Training Monitor** | Live run tracking backed by the middleware's tracking database. |
| **Item Database Browser** | Searchable, filterable item catalog view. |
| **Native Desktop Shell** | Single Rust/Tauri binary per platform (Linux `.deb`/`.AppImage`, macOS `.dmg`, Windows `.msi`). |

### IDE & Engine Integrations

[![VS Code](https://img.shields.io/badge/VS_Code-Extension-007ACC?logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/) [![Unreal Engine](https://img.shields.io/badge/Unreal_Engine-Plugin_(planned)-0E1128?logo=unrealengine&logoColor=white)](https://www.unrealengine.com/)

| Capability | Description |
| --- | --- |
| **VS Code Extension** (`extension/`) | Authoring support, schema validation, and inline previews for Hydra `game`/`policy` YAML configs. |
| **Unreal Engine Plugin** (planned) | In-editor build optimization for UE-based game projects; tracked on the [project board](#documentation). |

## Architecture

```
Build-Optimization/
├── main.py                     # Hydra entry point
├── backend/                    # C++ solver core (pybind11 extension)
│   ├── CMakeLists.txt
│   ├── pixi.toml
│   ├── include/
│   └── src/
├── middleware/                 # Python orchestration layer
│   ├── pyproject.toml
│   ├── configs/                # Hydra configs (game, optimization, pipeline, policy)
│   ├── src/
│   │   ├── core/                 # Item, Build, Synergy, Scoring domain model
│   │   ├── policies/              # 28 WSmart-Route-derived metaheuristics
│   │   ├── pipeline/               # Data ingestion + transforms
│   │   └── tracking/               # Experiment tracking database
│   ├── ui/                     # Streamlit control tower dashboard
│   ├── tests/
│   └── validation/              # Static analysis tooling
├── app/                        # Tauri + TypeScript desktop Studio
│   ├── package.json
│   └── src-tauri/               # Rust shell (Cargo workspace member)
├── extension/                   # VS Code extension
│   └── package.json
├── research/                    # Domain research (knapsack-routing theory)
├── tools/*/justfile             # Per-domain command runner modules
├── .agent/                      # AI coding-assistant guide (AGENTS.md, prompts, rules, skills, workflows)
├── .github/                     # CI/CD workflows, CONTRIBUTING.md, codecov.yaml
├── .moon/                       # ROADMAP.md, CHANGELOG.md
├── docker/                      # Container build definitions
├── docs/                        # Architecture / development / testing references
├── env/                         # Conda environment + env-var templates
├── desktop/                     # Linux / macOS / Windows build & run scripts
├── package.json                 # Root NPM workspace (app, extension)
└── Cargo.toml                   # Root Cargo workspace (app/src-tauri, future Rust modules)
```

## Domain Mapping

| WSmart-Route (VRP)   | Build-Optimization            |
| --------------------- | ------------------------------ |
| Bins/Nodes             | Items/Equipment                |
| Vehicle routes         | Equipped build (slot → item)   |
| Vehicle capacity       | Budget constraint               |
| Distance cost           | Item cost                       |
| Collection profit       | Effectiveness score             |
| Destroy operators        | Remove items from build         |
| Repair operators          | Fill empty slots                |

## Available Solvers

| Solver                        | Key      | Description                           |
| ----------------------------- | -------- | -------------------------------------- |
| Greedy                        | `greedy` | Greedy fill by best score improvement  |
| Random Search                 | `random` | Best of N random builds                |
| Simulated Annealing           | `sa`     | Temperature-based acceptance            |
| Genetic Algorithm             | `ga`     | Crossover + mutation evolution          |
| Iterated Local Search         | `ils`    | Perturbation + local search restarts    |
| Late Acceptance Hill Climbing | `lahc`   | Score history-based acceptance          |
| Record-to-Record Travel       | `rrt`    | Record + tolerance threshold            |
| Guided Local Search           | `gls`    | Augmented cost with penalties           |
| Reactive Tabu Search          | `rts`    | Adaptive tabu tenure                    |
| Old Bachelor Acceptance       | `oba`    | Age-based acceptance threshold          |
| ALNS                          | `alns`   | Adaptive operator weighting             |

Plus the full `middleware/src/policies/` folder — 28 additional metaheuristic implementations carried over from WSmart-Route.

## Installation & Setup

### Prerequisites

| Component | Minimum | Purpose |
| --- | --- | --- |
| Python | 3.9+ | Middleware, solvers, dashboard |
| Node.js | 18+ | Tauri frontend, VS Code extension |
| Rust | 1.75+ | Tauri desktop shell |
| CMake | 3.18+ | C++ backend build |
| [uv](https://github.com/astral-sh/uv) | latest | Python dependency management |
| [Pixi](https://pixi.sh/) | latest | C++ toolchain / native dependency management |
| [Just](https://github.com/casey/just) | latest | Task runner used across all `justfile`s |

### 1. Python Middleware

```bash
# From the repo root — resolves the root pyproject.toml
uv sync

# Or scoped to the middleware package directly
cd middleware && uv sync
```

### 2. C++ Backend

```bash
cd backend
pixi install
just build-base   # configures + builds the pybind11 extension via CMake
```

### 3. Tauri Frontend & VS Code Extension (NPM Workspace)

```bash
# From the repo root — installs app/ and extension/ together
npm install

# Rust side of the Tauri shell
cd app/src-tauri && cargo fetch
```

### 4. Everything at once

```bash
just setup
```

## Running the Project

```bash
# Run optimization with default settings (SA solver, RPG game)
uv run python main.py

# Use a specific policy / game
uv run python main.py policy=policy_sa game=rpg

# Adjust budget and time limit
uv run python main.py policy=policy_alns optimization.budget=2000 optimization.time_limit=120

# Launch the Streamlit control tower dashboard
just dashboard

# Launch the Tauri desktop Studio (dev mode)
cd app && npm run tauri dev

# Run the VS Code extension in a dev host
cd extension && npm run watch   # then F5 in VS Code
```

See the root [`justfile`](justfile) for the full set of `train` / `eval` / `optimize` / `gui` / `dashboard` recipes.

## Testing

```bash
# Python middleware
uv run pytest middleware/tests -v

# C++ backend
cd backend && pixi run test

# Tauri frontend
cd app && npm test

# VS Code extension
cd extension && npm test
```

Coverage is tracked via [Codecov](https://codecov.io/) (config: [`.github/codecov.yaml`](.github/codecov.yaml)); the Python suite targets ≥60% (`pyproject.toml` → `[tool.coverage.report]`).

## Documentation

| Resource | Description |
| --- | --- |
| [`.agent/AGENTS.md`](.agent/AGENTS.md) | Guide for AI coding assistants working in this repo. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module boundaries and data flow. |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Environment setup and day-to-day workflows. |
| [`docs/TESTING.md`](docs/TESTING.md) | Test organization and coverage requirements. |
| [`.moon/ROADMAP.md`](.moon/ROADMAP.md) | Planned work, phased by module. |
| [`.moon/CHANGELOG.md`](.moon/CHANGELOG.md) | Release history. |
| [GitHub Project Board](https://github.com/users/ACFHarbinger/projects/15/) | Live issue tracking across C++ Backend + Python Middleware, IDE Extension, Unreal Engine Plugin, and Tauri App views. |

## Contributing

See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) for code style, git workflow, and the pull-request process.

## License

MIT — see [LICENSE](LICENSE).
