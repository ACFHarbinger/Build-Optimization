# Build-Optimization Roadmap

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)

> **Version**: 1.0
> **Date**: 2026-07-29
> **Status**: In Progress

## Overview

This document tracks planned implementation work for Build-Optimization, organized into the four tracks that mirror the [GitHub Project Board](https://github.com/users/ACFHarbinger/projects/15/) views: **C++ Backend + Python Middleware**, **IDE Extension**, **Unreal Engine Plugin**, and **Tauri App**. Completed items move to [`.moon/CHANGELOG.md`](CHANGELOG.md).

Status markers: ✅ Done · 🚧 In Progress · 📋 Pending

---

## Track: C++ Backend + Python Middleware

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| B1 | Scaffold `backend/` C++ module (`CMakeLists.txt`, `pixi.toml`, pybind11 bindings entry point) | S | 🚧 In Progress |
| B2 | Implement branch-and-bound exact solver for the 0-1 / multiple-choice knapsack | M | 📋 Pending |
| B3 | Implement quadratic knapsack solver for synergy / set-bonus terms | M | 📋 Pending |
| B4 | Add McCormick-envelope / Fortet-inequality linearization utilities for multiplicative stat terms | M | 📋 Pending |
| B5 | Expose backend solvers to Python via `pybind11` and wire them into `middleware/src/policies` as selectable Hydra policies | M | 📋 Pending |
| B6 | Implement lexicographic goal-programming solve loop (ranked objective sequencing) | L | 📋 Pending |
| B7 | Implement Benders decomposition solver for black-box simulator constraints | L | 📋 Pending |
| B8 | Add concrete `GameAPISource` and `WebScraperSource` pipeline implementations (currently skeletons) | M | 📋 Pending |
| B9 | Add GFlowNet-based amortized inference policy for build sampling | L | 📋 Pending |
| B10 | Add PPO/MCTS real-time itemization policy for MOBA-style adaptive builds | L | 📋 Pending |
| B11 | Design and implement the experiment tracking database schema (`middleware/src/tracking`) | M | 📋 Pending |
| B12 | Wire C++ build + Python test suite into `.github/workflows/ci.yml` | S | 📋 Pending |

## Track: IDE Extension

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| E1 | Scaffold `extension/` VS Code extension project (`package.json`, activation events, command palette entries) | S | 🚧 In Progress |
| E2 | Implement JSON Schema validation for Hydra `game` / `policy` / `optimization` YAML configs | M | 📋 Pending |
| E3 | Add inline autocomplete for solver keys, game profiles, and policy parameters | M | 📋 Pending |
| E4 | Add a "Run Optimization" editor command that shells out to `main.py` with the active config | S | 📋 Pending |
| E5 | Add build-result preview panel (webview rendering the last optimization result) | M | 📋 Pending |
| E6 | Publish the extension to the VS Code Marketplace and wire a release workflow | S | 📋 Pending |

## Track: Unreal Engine Plugin

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| U1 | Scaffold the UE plugin skeleton (`.uplugin` descriptor, module source layout) | S | 📋 Pending |
| U2 | Define C++ data marshalling between UE types and `Item` / `Build` / `Synergy` domain objects | M | 📋 Pending |
| U3 | Build an in-editor build-optimization panel (Slate UI) | L | 📋 Pending |
| U4 | Expose solver calls as Blueprint-callable nodes | M | 📋 Pending |
| U5 | Bridge the plugin to the C++ backend solvers (shared `backend/` core, no Python dependency at runtime) | L | 📋 Pending |
| U6 | Package and submit the plugin to Fab (Unreal Marketplace) | M | 📋 Pending |

## Track: Tauri App

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| T1 | Scaffold `app/` Tauri + React + TypeScript project (`package.json`, `src-tauri/`) | S | 🚧 In Progress |
| T2 | Implement the Build Explorer page (native port of `middleware/ui/pages/build_explorer.py`) | M | 📋 Pending |
| T3 | Implement the Solver Comparison page (native port of `middleware/ui/pages/solver_comparison.py`) | M | 📋 Pending |
| T4 | Implement the Training Monitor page backed by the middleware tracking database | M | 📋 Pending |
| T5 | Implement the Item Database Browser page | M | 📋 Pending |
| T6 | Wire Tauri Rust commands to the middleware's SQLite tracking database via `sqlx` | M | 📋 Pending |
| T7 | Add cross-platform bundling CI (Linux `.deb`/`.AppImage`, macOS `.dmg`, Windows `.msi`) | M | 📋 Pending |

---

## How to Use This Document

- Pick up any `📋 Pending` item, open a matching GitHub issue on the [project board](https://github.com/users/ACFHarbinger/projects/15/) under the track's view, and reference it in your PR.
- Mark items `🚧 In Progress` while active, and move completed entries into [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased` when merged.
- See [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md) for the full contribution workflow.
