# Build-Optimization Roadmap

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)

> **Version**: 1.0
> **Date**: 2026-07-29
> **Status**: In Progress

## Overview

This document tracks planned implementation work for Build-Optimization, organized into the four tracks that mirror the component labels on the [GitHub Project Board](https://github.com/users/ACFHarbinger/projects/15/): **C++ Backend + Python Middleware**, **Browser Extension**, **Unreal Engine Plugin**, and **Tauri App**. Completed items move to [`moon/CHANGELOG.md`](CHANGELOG.md).

Status markers: ✅ Done · 🚧 In Progress · 📋 Pending

---

## Track: C++ Backend + Python Middleware

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| B1 | Scaffold `backend/` C++ module (`CMakeLists.txt`, `pixi.toml`, pybind11 bindings entry point) | S | ✅ Done |
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
| B12 | Wire C++ build + Python test suite into `.github/workflows/ci.yml` | S | ✅ Done |

## Track: Browser Extension

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| E1 | Scaffold `extension/` Manifest V3 browser extension project (`package.json`, `manifest.json`, per-browser webpack configs) | S | ✅ Done |
| E2 | Add per-site wiki selector profiles (Fandom, wiki.gg, Gamepedia) with slot/rarity/stat-block mapping | M | 📋 Pending |
| E3 | Implement a review/edit UI in the popup for correcting scraped fields before export | M | 📋 Pending |
| E4 | Wire `EXPORT_ITEMS` output directly into `middleware/src/pipeline/file_source.py`'s expected JSON schema, with per-game presets | S | 📋 Pending |
| E5 | Add bulk-scrape support (category/index page crawling within a tab, rate-limited) | M | 📋 Pending |
| E6 | Package and publish to the Chrome Web Store and Firefox Add-ons, and wire a release workflow | S | 📋 Pending |

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
| T1 | Scaffold `frontend/` Tauri + React + TypeScript project (`package.json`, `src-tauri/`) | S | ✅ Done |
| T2 | Implement the Build Explorer page (native port of `middleware/ui/pages/build_explorer.py`) | M | 📋 Pending |
| T3 | Implement the Solver Comparison page (native port of `middleware/ui/pages/solver_comparison.py`) | M | 📋 Pending |
| T4 | Implement the Training Monitor page backed by the middleware tracking database | M | 📋 Pending |
| T5 | Implement the Item Database Browser page | M | 📋 Pending |
| T6 | Wire Tauri Rust commands to the middleware's SQLite tracking database via `sqlx` | M | 📋 Pending |
| T7 | Add cross-platform bundling CI (Linux `.deb`/`.AppImage`, macOS `.dmg`, Windows `.msi`) | M | 📋 Pending |

---

## How to Use This Document

- Pick up any `📋 Pending` item, open a matching GitHub issue labeled for the track on the [project board](https://github.com/users/ACFHarbinger/projects/15/), and reference it in your PR.
- Mark items `🚧 In Progress` while active, and move completed entries into [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased` when merged.
- See [`git/CONTRIBUTING.md`](../git/CONTRIBUTING.md) for the full contribution workflow.
