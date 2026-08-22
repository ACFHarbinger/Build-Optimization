# AGENT_BUS — index

Async, file-based coordination log for the agents working this repo
(Claude, opencode, grok, codex, agy) — same convention used on this
account's other multi-agent repos (Image-Toolkit, Online-Price-Comparator).
No shared runtime session — each agent reads/writes these files
independently, so treat them as the source of truth for "who's doing
what" and "what's actually landed," not `moon/ROADMAP.md`'s own status
column or the GitHub Project Board (both go stale — see the 2026-08-22
kickoff entry for a concrete example: 8 issues marked open on the board
for work that had already shipped).

**Post new entries to today's file:** `.agent/bus/<YYYY-MM-DD>.md`
(create it if today doesn't have one yet, heading convention:
`### <agent> — YYYY-MM-DD (topic)`).

**House rules (read before posting):**

1. **Verify against real code before claiming something is done or
   available to build on.** Read the file, run the test, don't trust
   `moon/ROADMAP.md`'s status column or a GitHub issue's open/closed state
   at face value — both have been found stale here already (day-one
   kickoff closed 8 issues for already-shipped V1-V8 work). This repo is
   a polyglot monorepo (Python/C++/TypeScript/Rust) — "verify" means
   actually running that layer's own test command, not just reading code.
2. **Claim disjoint files/modules before starting** — post which
   package/track you're taking so two agents don't collide. This repo's
   own module boundaries (`.agent/AGENTS.md` §3) are a good default
   split: `middleware/src/core` (domain model) is upstream of
   `middleware/src/solvers`/`policies`; `backend/` (C++) is upstream of
   both; `frontend/` and `extension/` are independent leaves with no
   runtime dependency on each other or on `backend/`. If someone else
   already claimed a track, reply here before doing conflicting work.
3. **When you land something**, post: commit hash, what you verified it
   with (name the exact command per layer — `pytest`, `pixi run test`,
   `npm test`, `cargo test` — and the count/result), and **what the next
   person's code should assume**. Update `moon/CHANGELOG.md` and
   `moon/ROADMAP.md`'s status marker for that row in the same commit —
   move it from 📋 Pending to ✅ Done (or 🚧 In Progress if partial).
4. **`gh` is set up and authenticated.** Real GitHub issues exist, one
   per roadmap row, numbered, labeled by component
   (`component:backend-middleware` / `component:browser-extension` /
   `component:unreal-plugin` / `component:tauri-app`) — see the
   [Project Board](https://github.com/users/ACFHarbinger/projects/15/).
   Reference the issue number when you claim/land something (`B13 / #32`).
   Claude handles issue creation/closing/comments by default; if you have
   `gh` access yourself, posting your own landing comment directly on the
   issue in addition to this bus log is fine.
5. **Harbinger** (pkhunter, the user) is the human sign-off role. Don't
   invent an agent identity for that; flag anything needing a product/
   architecture decision by name, addressed to Harbinger, same as the
   other repos' convention.
6. **No test-execution caution active** — nothing analogous to
   Online-Price-Comparator's old CPU-cooler issue has been raised for
   this machine. Full test suite runs (all four layers) are fine.
7. **Known real gap, not a suggestion to fix silently**: B13 (`#32`) — 51
   pre-existing `mypy` errors across 22 files mean `lint-python` in CI has
   likely never been green. Don't let a later "all checks passed" claim
   on unrelated work imply this is fixed unless you actually ran
   `uv run mypy src` yourself and it's clean.

**Reading history:** the current/most-recent day lives under
`.agent/bus/`; once a day stops being "today," move it to
`.agent/archive/bus/<date>.md` and start a fresh dated file.

| Day | Location |
|---|---|
| 2026-08-22 (current) | `.agent/bus/2026-08-22.md` |
