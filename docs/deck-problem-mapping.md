# Slay the Spire 2 Domain Mapping: Cards to Items & Deck to Knapsack Subset

> **Version**: 1.0  
> **Date**: 2026-07-30  
> **Track**: Slay the Spire 2 Vertical Slice (V1)

---

## 1. Domain Mismatch: Equipment Builds vs. Deckbuilding

In traditional equipment-based RPG/MOBA/Darktide build optimization:
- Each build consists of **exactly one item per fixed equipment slot** (e.g. `WEAPON`, `HELMET`, `CHEST`).
- Slot choices are mutually exclusive (MCKP: Multiple-Choice Knapsack Problem).
- Optimization searches over item combinations where each slot receives 1 item from its candidate pool.

In **Slay the Spire 2** (and deckbuilding games generally):
- A deck is a **variable-size subset** of cards selected from a pool.
- There are **no slot exclusivity constraints**: a deck may contain any number of `ATTACK`, `SKILL`, or `POWER` cards.
- The deck is bounded by a target deck size cap ($K$, e.g. 16–18 cards for Act 2) and an optional gold/energy budget limit.
- This maps directly to a **0-1 Knapsack Problem** (weight = 1 per card, capacity = `max_deck_size`, value = archetype score contribution).

---

## 2. Field-by-Field Domain Mapping

| Slay the Spire 2 Concept | Build-Optimization Domain Model | Notes |
| --- | --- | --- |
| **Card** | `core.item.Item` | Unique ID, display name, cost, rarity, stats dict, and tag set. |
| **Card Type** (`Attack`, `Skill`, `Power`) | `core.item.Slot` (`Slot.ATTACK`, `Slot.SKILL`, `Slot.POWER`) | Reuses the existing `Slot` enum for display/taxonomy; not used for slot-exclusivity in decks. |
| **Energy / Gold Cost** | `Item.cost` | Card energy cost (or acquisition gold cost). |
| **Card Rarity** (`Common`, `Uncommon`, `Rare`) | `core.item.Rarity` | Tier multipliers (`COMMON` = 1.0, `UNCOMMON` = 1.25, `RARE` = 1.5, etc.). |
| **Card Power / Stat Contributions** | `Item.stats` | Numeric dict (e.g., `damage`, `block`, `strength_gen`, `multi_hit_count`). |
| **Archetype Tags** (`strength`, `multi_hit`, `scaling`) | `Item.tags` | Set of strings used for Synergy rules and archetype matching. |
| **Deck** | `core.deck.Deck` | Subclasses `core.build.Build`, representing a list of selected cards. |
| **Target Deck Size** | `DeckProblem.max_deck_size` | Knapsack capacity constraint ($K$). |
| **Archetype Synergies** | `core.synergy.SynergyEngine` | Rules rewarding card combinations (e.g., Strength Scaling + Multi-Hit Attacks). |

---

## 3. Architecture Parity & Interoperability

Because `Deck` subclasses `Build` and implements `equipped_items`:
1. `core.scoring.score_build` works on a `Deck` without modifications.
2. `SynergyEngine.active_synergies` works on a `Deck` without modifications.
3. `pipeline.games.optimizer._persist_run` and `_build_to_result_json` serialize `Deck` results into the standard JSON schema expected by the Tauri Studio frontend (`frontend/`) and tracking SQLite database (`tracking.db`).

---

## 4. Solvers & Execution Flow

- **C++ Backend**: Reuses the exact 0-1 knapsack solver (`backend/src/knapsack.cpp` / `solve_knapsack` pybind11 binding).
- **Python Fallback**: Gracefully falls back to deterministic greedy-by-value selection if the C++ backend binary is not compiled.
- **Hydra Configuration**: Managed via `game=slay_the_spire_2` and `policy=policy_deck_knapsack` or `policy=policy_deck_greedy`.
