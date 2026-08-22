# Handoff: Build-Optimization multi-agent bus kickoff

**Date**: 2026-08-22
**From**: Claude, same session that ran the multi-agent workflow on
Online-Price-Comparator for the prior two days, now pivoted here at
Harbinger's request.
**Purpose**: enough context for a fresh Claude session (new context
window, no memory of this session) to pick up the coordination role on
this repo without re-deriving the working pattern from scratch.

---

## 1. What this session's role actually is

Harbinger runs several coding-agent CLIs in parallel terminals against
this repo — `grok`, `codex`, `agy` (Gemini), `opencode` — each a real,
separately-billed session with its own model and its own read/write
access to this working tree. Claude's (your) role is **coordinator, not
sole implementer**: read the repo, reconcile what's actually true against
what the docs/issue-tracker claim, post a delegation to the bus, and when
Harbinger says the round is done, verify what landed, reconcile
`moon/CHANGELOG.md`/`moon/ROADMAP.md`/GitHub issues, and post the next
round. You *can* implement small things directly (a stale doc line, a
one-line fix found while verifying), but the bulk of code changes come
from the other four agents picking up bus posts.

**The operating loop, verbatim, repeats indefinitely:**

1. Harbinger says something like *"All agents have finished, let us
   proceed"* (or just *"Let us proceed"*).
2. You run each layer's real verification command (see §4) and `git log
   --oneline` / `git status` to see what actually landed since your last
   check — **never trust a bus claim or a roadmap status marker without
   running the check yourself.**
3. Read the new bus entries (claims + landings) to understand what each
   agent did and why.
4. Reconcile on GitHub: comment/close issues for what's genuinely done,
   leave a progress comment (issue stays open) for partial work, and fix
   any stale `moon/ROADMAP.md`/`moon/CHANGELOG.md` lines yourself if the
   landing agent didn't.
5. **If a delegated track never started at all** (no claim, no commit,
   nothing) — this happens periodically, don't assume it's an error, just
   re-post it, usually with a narrower scope than the first attempt. Don't
   ask Harbinger why; just retry once, and if a specific track stalls
   *twice*, say so explicitly in the bus post rather than posting a third
   identical retry.
6. Post the next round: **four tracks, one per agent, on genuinely
   disjoint files/modules** so nobody's edits collide. Prefer continuing
   an agent's own existing thread (e.g. whoever built the extension
   scaffolding keeps extending it) over reassigning cold every round —
   agents build real context within their own thread across rounds.
7. Repeat from step 1 on the next "let us proceed."

**When something bigger than routine delegation comes up** (a real
architecture/policy decision, an ambiguous instruction with high cost of
guessing wrong), use `AskUserQuestion` rather than assuming — this
happened several times on Online-Price-Comparator (a scope-expansion
decision, a legal/ToS question) and Harbinger engaged with it directly
each time rather than being annoyed by the pause.

## 2. Real gaps found this way, worth knowing the *pattern*, not just the specific gap

On Online-Price-Comparator, this loop caught real problems that a purely
code-generation approach wouldn't have:

- **Stale-in-both-directions status tracking**: roadmap docs and GitHub
  issues drift out of sync with actual code in *both* directions — some
  "Done" rows were still missing real pieces, some "Planned"/open-issue
  rows were actually fully shipped. Always verify by reading the file and
  running the test, not by reading the status marker.
- **A policy documented but not enforced in code**: a roadmap doc said
  "site X is not scraped server-side" but nothing in the code actually
  checked for that — the mechanism existed only as a comment/decision
  record, not as a runtime guard. Same class of bug already found once on
  this repo (see the bus kickoff entry — 8 issues open for shipped work,
  the *opposite* direction of the same "don't trust the tracker" lesson).
  **When reviewing landed work, specifically ask "is this constraint
  enforced in code, or only documented?"** — a working test suite doesn't
  catch a missing guard nobody wrote a test for.
- **Concurrent commits in a shared working tree can sweep each other's
  staged-but-uncommitted files** into the wrong commit. Content is never
  actually lost (files persist on disk regardless of git state), but
  attribution/commit messages can end up misleading. Agents on this
  account have started self-documenting this ("my staged X got swept into
  Y's commit, noting it here") — that's the right response, not a bug to
  chase further.

## 3. This repo specifically: what's real, what's not, as of 2026-08-22

- **Stack**: Python 3.9+ middleware (`uv`), C++17 backend (`pixi`/CMake,
  `pybind11`), Tauri 2.0 + React 19 + TypeScript 5 frontend, Rust (Tauri
  shell only), a Manifest V3 browser extension, and a *planned* (not yet
  started beyond scaffolding-in-progress) Unreal Engine plugin. Full
  reference: `.agent/AGENTS.md` (module boundaries, CLI entry points,
  coding standards, known constraints) and `README.md` (architecture
  diagram, domain mapping, available solvers).
- **Roadmap**: `moon/ROADMAP.md`, five tracks (C++ Backend + Python
  Middleware, Browser Extension, Unreal Engine Plugin, Tauri App,
  Documentation) plus a completed cross-cutting "Slay the Spire 2
  Vertical Slice" track. GitHub Project Board:
  https://github.com/users/ACFHarbinger/projects/15/, issues labeled
  `component:backend-middleware` / `component:browser-extension` /
  `component:unreal-plugin` / `component:tauri-app`.
- **What happened today (2026-08-22), before this handoff was written**:
  - Set up `.agent/bus/` (this was the first bus-convention setup on this
    repo — read `.agent/bus/AGENT_BUS.md` for the full house rules,
    they're not repeated here).
  - Closed 8 stale-but-shipped issues (#33-#40, Slay the Spire 2 V1-V8) —
    see §2's second bullet for why this matters as a pattern, not just a
    one-off cleanup.
  - Posted the **first delegation round** (`.agent/bus/2026-08-22.md`):
    **B13/#32** (Grok — fix 51 pre-existing `mypy` errors blocking CI),
    **E2/#14** (Codex — per-site wiki selector profiles for the browser
    extension), **T7/#31** (Agy — cross-platform Tauri bundling CI),
    **U1/#19** (opencode — scaffold the Unreal Engine plugin skeleton).
  - **None of these four have been checked on yet** — this handoff was
    written immediately after posting the round, before any "let us
    proceed." Whoever picks this up next should treat step 2 of the loop
    in §1 as the very next action: check what landed, don't assume
    anything from this round happened yet.

## 4. Per-layer verification commands (use the real ones, not a guess)

| Layer | Command | Notes |
|---|---|---|
| Python middleware | `cd middleware && uv run pytest tests -v` | Coverage gate 60% (`pyproject.toml`). |
| Python lint/type | `cd middleware && uv run mypy src` | **Currently red** — 51 known errors, B13/#32 is the fix-it track. Don't let an unrelated landing claim "clean" without actually running this. |
| C++ backend | `cd backend && pixi run test` | Needs `pixi install` + `pixi run build` first if not already built. |
| Tauri frontend | `cd frontend && npm test` | |
| Browser extension | `cd extension && npm test` | |
| Full local setup | `just setup` (repo root) | Syncs all four environments. |

## 5. Where to look, not what to memorize

Don't try to hold the full roadmap/architecture in your head from this
handoff alone — it will drift the moment more work lands. Always re-read
`.agent/bus/AGENT_BUS.md` (house rules), the current day's bus file
(what's actually claimed/landed), `moon/ROADMAP.md` (current status
markers — verify, don't trust), and `gh issue list --repo
ACFHarbinger/Build-Optimization --state open` before making any claim
about repo state. This handoff is a map of *the process*, not a snapshot
of *the content* — the content will already be stale by the time someone
reads this.
