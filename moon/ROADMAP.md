# Build-Optimization Roadmap

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24C8DB?logo=tauri&logoColor=white)](https://tauri.app/)

> **Version**: 1.0
> **Date**: 2026-07-29
> **Status**: In Progress

## Overview

This document tracks planned implementation work for Build-Optimization. The first four tracks mirror the component labels on the [GitHub Project Board](https://github.com/users/ACFHarbinger/projects/15/): **C++ Backend + Python Middleware**, **Browser Extension**, **Unreal Engine Plugin**, and **Tauri App**. **Documentation** and **Slay the Spire 2 Vertical Slice** are cross-cutting tracks without a dedicated board label — their issues are tagged with whichever component label matches where the work actually lives. Completed items move to [`moon/CHANGELOG.md`](CHANGELOG.md).

Status markers: ✅ Done · 🚧 In Progress · 📋 Pending

---

## Track: C++ Backend + Python Middleware

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| B1 | Scaffold `backend/` C++ module (`CMakeLists.txt`, `pixi.toml`, pybind11 bindings entry point) | S | ✅ Done |
| B2 | Implement branch-and-bound exact solver for the 0-1 / multiple-choice knapsack | M | ✅ Done |
| B3 | Implement quadratic knapsack solver for synergy / set-bonus terms | M | 📋 Pending |
| B4 | Add McCormick-envelope / Fortet-inequality linearization utilities for multiplicative stat terms | M | 📋 Pending |
| B5 | Expose backend solvers to Python via `pybind11` and wire them into a selectable Hydra policy (`policy=policy_bnb`) | M | ✅ Done |
| B6 | Implement lexicographic goal-programming solve loop (ranked objective sequencing) | L | 📋 Pending |
| B7 | Implement Benders decomposition solver for black-box simulator constraints | L | 📋 Pending |
| B8 | Add concrete `GameAPISource` and `WebScraperSource` pipeline implementations (currently skeletons) | M | 📋 Pending |
| B9 | Add GFlowNet-based amortized inference policy for build sampling | L | 📋 Pending |
| B10 | Add PPO/MCTS real-time itemization policy for MOBA-style adaptive builds | L | 📋 Pending |
| B11 | Design and implement the experiment tracking database schema (`middleware/src/tracking`) | M | ✅ Done |
| B12 | Wire C++ build + Python test suite into `.github/workflows/ci.yml` | S | ✅ Done |
| B13 | Fix the pre-existing `uv run mypy src` failures (51 errors in 22 files, all inherited RL-pipeline code — `configs.Config` attribute errors, torch `Tensor \| Module` union-attr issues, etc.); `lint-python` in CI has likely never been green | M | 📋 Pending |

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
| T2 | Implement the Build Explorer page (native port of the archived `middleware/ui` Streamlit page) | M | ✅ Done |
| T3 | Implement the Solver Comparison page (native port of the archived `middleware/ui` Streamlit page) | M | ✅ Done |
| T4 | Implement the Training Monitor page (file-based: reads `outputs/**/metrics.csv` / `training_log.jsonl`) | M | ✅ Done |
| T5 | Implement the Item Database Browser page | M | ✅ Done |
| T6 | Wire Tauri Rust commands to the middleware's SQLite tracking database via `sqlx` | M | ✅ Done |
| T7 | Add cross-platform bundling CI (Linux `.deb`/`.AppImage`, macOS `.dmg`, Windows `.msi`) | M | ✅ Done |
| T8 | Code-split the ECharts bundle (currently a single >1.3MB chunk) via dynamic `import()` per page | S | 📋 Pending |
| T9 | Replace Training Monitor's/Item Database's file-based reads (T4/T5) with the T6 tracking-DB commands now that they exist; extend `TrackedRunsPanel` (currently only in Build Explorer) to Solver Comparison | M | 📋 Pending |

## Track: Documentation

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| D1 | Scaffold the MkDocs Material portal (`docs/mkdocs.yml`) embedding README/CONTRIBUTING/ROADMAP/CHANGELOG via `pymdownx.snippets` | S | ✅ Done |
| D2 | Wire sphinx-autoapi for the Python middleware reference (`docs/sphinx/`) | S | ✅ Done |
| D3 | Wire Doxygen for the C++ backend reference (`docs/cpp/Doxyfile`) | S | ✅ Done |
| D4 | Wire TypeDoc for the Tauri frontend and browser extension references | S | ✅ Done |
| D5 | Wire `cargo doc` (rustdoc) for the Tauri Rust shell reference | S | ✅ Done |
| D6 | Orchestrate all five generators + MkDocs in `docs/build_docs.sh`, tolerant of missing tools/unscaffolded modules | S | ✅ Done |
| D7 | Wire `.github/workflows/docs.yml` (per-generator jobs + GitHub Pages deploy on `main`) | S | ✅ Done |
| D8 | Add a `docs/plugin/` Unreal Engine plugin Doxygen config once the plugin (U1) is scaffolded | S | 📋 Pending |

## Track: Slay the Spire 2 Vertical Slice

A cross-cutting initiative (no dedicated board label — issues are tagged `component:backend-middleware`/`component:tauri-app` per where the work lives), based on [`reports/Slay the Spire 2 Guide.md`](../reports/Slay%20the%20Spire%202%20Guide.md). Goal: a *thin, real, end-to-end* path — one character, one archetype — proving the domain generalizes beyond fixed-equipment-slot games, not a full simulation of the game's mechanics.

**Domain mismatch this track has to resolve**: every existing game (`rpg`/`moba`/`darktide`) is modeled as *exactly one item per fixed equipment slot* (`core.item.Slot`, `core.build.Build`). Slay the Spire 2 deckbuilding has no slots — a deck is a *variable-size subset* of cards from a pool, bounded by a target deck size (the report's own strategic lever: 6-8 cards early Act 1, 16-18 by Act 2, 20-30 in Act 3, or a compressed 10-15 for the Regent's Star Engine). That's a plain 0-1 knapsack (weight=1 per card, capacity=target deck size, value=an archetype-fit score) — **not** the Multiple-Choice Knapsack B2/B5 just wired up (which requires mutually-exclusive classes/slots). The existing `solve_knapsack` C++ binding (built in B2 alongside the MCKP solver, unused since B5 chose MCKP for equipment) is exactly this shape, so **the vertical slice needs zero new C++ code** — pure Python/config wiring, reusing `backend/`'s existing plain-knapsack export.

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| V1 | Design doc: the Card~Item / Deck~knapsack-subset mapping (this table's second paragraph, expanded with field-by-field mapping) as a short `docs/` note, so V2-V8 have one source of truth instead of re-deriving it | S | ✅ Done |
| V2 | Add `middleware/src/core/deck_problem.py`: a `DeckProblem` class parallel to `BuildProblem` but for *subset* selection (no per-slot exclusivity) — `score_fast`/`score_full`-equivalent methods, and a `to_result_json()` emitting the *same* `{solver, score, cost, budget, items, synergies}` shape the existing games use, so the current frontend pages and tracking pipeline (B11/T6) work on it completely unmodified | M | ✅ Done |
| V3 | Register a `deck` solver (new `middleware/src/pipeline/decks/` mini-pipeline, mirroring `pipeline/games`) that calls the existing `solve_knapsack` binding via `core.native_backend`, with a pure-Python greedy fallback when the backend isn't built (matching `bnb`'s graceful-failure precedent from B5) | M | ✅ Done |
| V4 | Author a sample Ironclad card dataset (~30-40 cards: Strikes/Defends plus the report's "Strength Scaling" archetype priorities — passive Strength generators, multi-hit attacks) as `middleware/src/data/sample/slay_the_spire_2_ironclad.json`, reusing the existing item-JSON schema (`name`/`slot`=card type/`stats`=power contributions/`cost`=energy/`rarity`/`tags`=archetype tags e.g. `strength`, `multi_hit`, `scaling`) | M | ✅ Done |
| V5 | Add `middleware/configs/game/slay_the_spire_2.yaml` (stat weights rewarding the Strength-scaling/multi-hit synergy from the report; deck-size "budget" targeting the Act-1-boss-to-Act-2 range of 16-18 cards) and a matching `middleware/configs/optimization/slay_the_spire_2.yaml` | S | ✅ Done |
| V6 | Wire `main.py`: add `slay_the_spire_2` to the game roster and `deck` to `_PIPELINE_SOLVERS`/policy resolution so `python main.py game=slay_the_spire_2 policy=policy_deck` runs end-to-end — result JSON to `outputs/`, run logged to the tracking DB, exactly like every other game (B11's `_persist_run` needs no changes) | S | ✅ Done |
| V7 | Add `middleware/tests/test_deck_problem.py`: a hand-verified small instance plus a brute-force cross-check, matching the rigor of B2's `test_mckp.cpp` | S | ✅ Done |
| V8 | Verify frontend rendering: confirm Build Explorer and Item Database Browser render an STS2 result/card-pool file correctly via the existing generic schema (expected: no frontend code changes; flag anything that reads oddly, e.g. the KPI row's "Budget Left" label for a deck-size-shaped budget) | S | ✅ Done |
| V9 | *(Explicitly beyond the vertical slice — future track)* Additional characters/archetypes (Silent Poison Engine, Regent Star Engine, Necrobinder Doom Execution, Defect Orb/Focus scaling), Enchantments and Ancients as scoring bonuses/penalties, Act-phased deck-size constraints, boss/elite encounter modeling | XL | 📋 Pending |

## Track: Slay the Spire 2 — Screenshot Reward Advisor

A cross-cutting initiative (no dedicated board label — issues tagged `component:backend-middleware`/`component:tauri-app` per where the work lives), built on top of the Vertical Slice track above. Scope is deliberately narrow: rank the post-fight 3-card reward choice — `{Skip, offer 1, offer 2, offer 3}` — for a confirmed Ironclad deck. Shop, relic/potion pick-ups, and mid-combat card-play are explicit future tracks, not this one. Full design discussion: [`.agent/bus/2026-08-22.md`](../.agent/bus/2026-08-22.md) (brainstorm with Grok/Codex/Agy/opencode).

**Not a knapsack — do not extend `DeckProblem`.** V1-V8's `DeckProblem`/`run_deck_optimization` answers "pick the best subset of N cards from a catalogue up to a size cap," and `score_build`'s `slot_bonus` term is a systematic *fill-the-deck* bias — the opposite of this feature's actual question, "is this offer worth a draw slot in an already-fixed deck?" (dilution must be able to make `Skip` win). SA3 below is a new, separate evaluator module that reuses only `core.scoring.score_build`/`SynergyEngine`, leaving `DeckProblem` and the existing `game=slay_the_spire_2` pool-subset path completely untouched.

**Recognition seam**: local OCR first, crop-box only (card names are art-integrated — whole-image OCR doesn't work), never the raw image sent to a cloud provider — only extracted text, and only once cloud fallback is explicitly enabled. Real STS2 screenshots and any extracted game assets are never committed to this public AGPL repo; only small deterministically-generated synthetic fixtures are. Unknown/ambiguous card names always block for manual correction — never a silent guess.

**Catalogue**: ingested from `slaythespire.wiki.gg`'s structured card data into a gitignored, user-local app-data cache (re-ingested on demand, since STS2 patches) with in-app + `THIRD_PARTY_NOTICES` attribution; the repo itself commits no scrape dump, only the existing small sample/synthetic catalogue used by tests. A local overlay (also gitignored) lets the user add cards missing from the cache or override a wiki row by `card_id` — the escape hatch when a card can't yet be recognized/ingested. `Strike`/`Strike+` etc. are distinct catalogue ids; display name is never the identity.

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| SA1 | Define the advisor's versioned JSON decision contract: deck (character-tagged multiset of card ids/counts), 3 offers, optional context (gold, relics, potions, Act/floor, HP%), preferences, catalogue/scoring-config versions, recognition diagnostics, ranked result + explanations | S | 📋 Pending |
| SA2 | Canonical stable-ID STS2 card catalogue: `slaythespire.wiki.gg` ingestion into a gitignored app-data cache (never committed) with attribution notices (committed), base/`+`-variant ids, aliases, character tag; a gitignored local overlay for user-added/overridden rows | M | 📋 Pending |
| SA3 | New `middleware/src/core/reward_eval.py`: pure `evaluate_reward(base, offers, scoring, synergies, preferences, context) -> RewardAdvice`, scoring `score_build(Deck(base))` vs `score_build(Deck(base + offer))` per action over a duplicate-preserving multiset — **not** a `DeckProblem`/knapsack change, and not using `score_fast`'s pool-relative normalization | M | 📋 Pending |
| SA4 | Multi-objective Pareto scoring over SA3's four outcomes: immediate tempo/survival, synergy/archetype delta, dilution/consistency (must let `Skip` win), resource/run-risk resilience — return the full non-dominated set, never one collapsed weighted number | L | 📋 Pending |
| SA5 | Seeded Monte-Carlo projected-run planner: samples remaining rarity-weighted reward offers (and shop buy-vs-remove if gold is provided) across the Act/floor horizon, no combat simulation; user-visible seed, configurable rollout count, mean + bands per `{Skip, A, B, C}` | L | 📋 Pending |
| SA6 | One-shot `advisor_cli.py` (versioned JSON in/out, app-controlled temp paths), Python core decoupled as `STS2AdvisorService` for a later local-daemon migration; manual/typed deck+offer+context entry (per-character starter deck, fork-starter, blank, named local presets) is the first working end-to-end input, shipped before recognition exists | M | 🚧 In Progress |
| SA7 | New top-level Tauri "Advisor" page (a study screen, not a combat overlay) invoking SA6 via `tauri-plugin-shell`: full Pareto frontier + MC bands, priority sliders/presets, comparison vs every offer and Skip, score/metric breakdowns, synergy/weakness explanations, deck-preset picker, optional context pickers | L | ✅ Done |
| SA8 | Local screenshot ingestion: clipboard paste + drag-drop + disk load, 16:9 and 16:10 (Steam Deck) crop-box variants; configured crop boxes locate the 3 offer regions and the deck-grid screen (no detection model this slice); crop each name-banner sub-region and OCR only that crop. Committed fixtures are a deterministic synthetic generator's output, never real screenshots | L | 📋 Pending |
| SA9 | Visual deck-grid reconstruction from the second screenshot type, with **mandatory** manual confirmation/editing of every detected card and count before the advisor may run — this is what makes V1 "screenshot-complete"; explicitly not in the first shippable milestone's critical path | M | 📋 Pending |
| SA10 | `RecognizedName` seam (`raw_text`, `normalized`, `confidence`, `method: exact\|alias\|fuzzy\|cloud\|manual`, `matched_card_id`, `needs_dataset_entry`) plus a per-user confidence threshold: at/above it, a flagged tentative answer; below it, block for correction. Unmatched/ambiguous names never silently resolve | M | 📋 Pending |
| SA11 | Optional cloud-fallback recognition, disabled by default, enabled only via explicit config + credential (`RECOGNITION_CLOUD_API_KEY`, absent ⇒ disabled): sends extracted text + candidates only, **never the screenshot image** in this phase | M | 📋 Pending |
| SA12 | Test the slice: unit tests for `evaluate_reward`/Pareto/MC planner (seeded/deterministic), synthetic OCR fixtures, Tauri interaction tests, a Skip/explanation regression suite, and a manual acceptance checklist for real screenshots (run locally, never committed) | L | 📋 Pending |

**Gates**: **G1** (ship first) SA1, SA3, SA4, SA5, SA6, SA7 — typed/manual deck+offer entry with dilution-aware Pareto scoring *and* seeded MC bands *and* the Advisor page, all before any screenshot recognition exists (dilution and MC are not deferred to a later gate). **G2** richer optional context pickers/preference presets not already covered by G1. **G3** SA8-SA10 — screenshot ingestion + mandatory confirmation (V1 complete per this feature's definition of done). **G4** SA11-SA12 — opt-in text-only cloud fallback + hardening.

---

## How to Use This Document

- Pick up any `📋 Pending` item, open a matching GitHub issue labeled for the track on the [project board](https://github.com/users/ACFHarbinger/projects/15/), and reference it in your PR.
- Mark items `🚧 In Progress` while active, and move completed entries into [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased` when merged.
- See [`git/CONTRIBUTING.md`](../git/CONTRIBUTING.md) for the full contribution workflow.
