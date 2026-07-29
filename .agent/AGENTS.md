# AGENTS.md - Instructions for Coding Assistant LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)

> **Version**: 1.0
> **Last Updated**: 2026-07-29
> **Purpose**: Authoritative reference for AI assistants (Claude, GPT, Copilot, etc.) working on Build-Optimization.

## Table of Contents

1. [Project Overview & Mission](#1-project-overview--mission)
2. [Technical Stack & Governance](#2-technical-stack--governance)
3. [Module Boundaries](#3-module-boundaries)
4. [Key CLI Entry Points](#4-key-cli-entry-points)
5. [Coding Standards](#5-coding-standards)
6. [Known Constraints](#6-known-constraints)

## 1. Project Overview & Mission

Build-Optimization solves videogame character "build" optimization — selecting items, skills, and stat allocations that maximize effectiveness under a resource budget — using techniques from combinatorial routing/knapsack research (see [`research/`](../research/)) adapted from the sibling project [WSmart-Route](https://github.com/ACFHarbinger/WSmart-Route).

## 2. Technical Stack & Governance

| Component | Specification | Notes |
| --- | --- | --- |
| Python | 3.9+ | Managed via `uv`; always `source .venv/bin/activate` |
| C++ | 17 | Built via CMake, environment managed by Pixi |
| TypeScript | 5 | Tauri frontend (`app/`) and VS Code extension (`extension/`) |
| Rust | stable | Tauri shell only (`app/src-tauri/`) |
| Config | Hydra | All solver/game/pipeline parameters composed from `middleware/configs/` |

## 3. Module Boundaries

- `middleware/src/core` — domain model (`Item`, `Build`, `Synergy`, `Scoring`). No imports from `ui/`, `backend/`, or `app/`.
- `middleware/src/solvers` / `middleware/src/policies` — algorithms. May call into `backend/`'s `pybind11` module but must not duplicate its logic in Python.
- `backend/` — C++ solver core. Exposes only what's declared in `backend/src/bindings.cpp`; no Python-specific logic leaks into C++.
- `app/` and `extension/` — presentation only. They read the tracking database or shell out to `main.py`; they never reimplement scoring/solving.

Full data-flow diagram: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## 4. Key CLI Entry Points

| Action | Command |
| --- | --- |
| Sync all environments | `just setup` |
| Run optimization | `uv run python main.py policy=policy_sa game=rpg` |
| Launch dashboard | `just dashboard` |
| Build C++ backend | `cd backend && pixi run build` |
| Launch Tauri Studio (dev) | `cd app && npm run tauri:dev` |
| Run Python tests | `uv run pytest middleware/tests -v` |
| Run C++ tests | `cd backend && pixi run test` |
| Run frontend tests | `cd app && npm test` |

## 5. Coding Standards

- Python: type hints on public functions, Google-style docstrings, Ruff for lint/format (see `middleware/pyproject.toml`).
- C++: header/implementation split under `backend/include/` and `backend/src/`, RAII, no raw owning pointers.
- TypeScript: functional React components, no default exports for shared utilities.
- Never hardcode credentials — use `env/vars.env` (gitignored; copy from `env/vars.env.example`).

## 6. Known Constraints

- The C++ backend has no Python-only fallback for its exact solvers — if `backend/` isn't built, only the pure-Python heuristics in `middleware/src/solvers` are available.
- The Tauri Studio (`app/`) reads the middleware's tracking database directly; it does not (yet) invoke Python at runtime.
- Coverage gate is 60% for the Python middleware; C++ and TypeScript modules are build/test-pass gated only (see `.github/codecov.yaml`).
