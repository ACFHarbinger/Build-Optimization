# Architecture

## Module Boundaries

```mermaid
flowchart TD
    subgraph Entry Points
        CLI[main.py — Hydra CLI]
        Studio[frontend/ — Tauri Studio]
        Extension[extension/ — Browser Extension]
    end

    subgraph Python Middleware
        Core[middleware/src/core — Item, Build, Synergy, Scoring]
        Solvers[middleware/src/solvers]
        Policies[middleware/src/policies — 28 metaheuristics]
        Pipeline[middleware/src/pipeline — data ingestion]
        Tracking[middleware/src/tracking — run/metric DB]
    end

    subgraph C++ Backend
        Backend[backend/ — pybind11 extension]
    end

    CLI --> Core
    CLI --> Solvers
    CLI --> Policies
    Studio -. reads outputs/ + data/ .-> Pipeline
    Extension -. exports item JSON .-> Pipeline
    Solvers --> Backend
    Policies --> Pipeline
    Pipeline --> Core
```

## Layering Rules

- `middleware/src/core` is the domain model and must not import from `backend/`.
- `backend/` exposes a stable `pybind11` API (`build_optimizer_backend`); Python code only calls into it through `core.native_backend.load_backend()` (see `pipeline/games/states/solving.py`'s `bnb` solver for the reference usage), never imports backend internals directly.
- `frontend/` (Tauri) is a presentation layer — its Rust commands (`frontend/src-tauri/src/commands/`) read solver-result and item-data files directly from `outputs/`/`data/` and shell out to `main.py`; it does not reimplement solver logic. Direct `middleware/src/tracking` database access is planned (see `moon/ROADMAP.md` item T6). `extension/` (browser extension) only produces item JSON for `middleware/src/pipeline`'s `FileSource` — it has no runtime dependency on the rest of the stack.

## Data Flow

1. Hydra composes `game` + `optimization` + `pipeline` + `policy` configs (`middleware/configs/`).
2. `middleware/src/pipeline` loads items via `FileSource`/`GameAPISource`/`WebScraperSource`.
3. The selected solver (native Python or `backend/`-bound C++) searches for the best build under `middleware/src/core.scoring`.
4. Results are written as JSON under `outputs/` and read directly by the Tauri Studio (`frontend/`); persistence via `middleware/src/tracking`'s database is wired into the CLI but not yet consumed by the Studio (moon/ROADMAP.md item T6).

See [`moon/ROADMAP.md`](../moon/ROADMAP.md) for planned architectural additions (lexicographic solve loop, Benders decomposition, GFlowNet policy).
