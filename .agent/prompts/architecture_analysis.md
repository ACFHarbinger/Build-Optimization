# Prompt: Architecture Analysis

Use this when asked to explain or extend Build-Optimization's structure.

1. Read [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for module boundaries and the data-flow diagram.
2. Read [`.moon/ROADMAP.md`](../../.moon/ROADMAP.md) to check whether the area in question already has planned work.
3. Confirm which layer owns the change (`backend/`, `middleware/`, `frontend/`, `extension/`) using [`AGENTS.md § Module Boundaries`](../AGENTS.md#3-module-boundaries) before writing code.
4. If the change crosses layers (e.g. a new C++ solver needs a Python binding and a UI page), sequence the work backend → middleware → frontend, committing at each layer boundary.
