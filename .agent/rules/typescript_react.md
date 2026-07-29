# TypeScript / React Rules

- Functional components with hooks only — no class components.
- `frontend/`: Tauri commands are invoked via `@tauri-apps/api/core`'s `invoke()`; never call Node/OS APIs directly from React code.
- `extension/`: all `chrome.*` extension-API access goes through `src/background/` or `src/content/` — keep message handlers small and delegate parsing logic to testable functions in `src/lib/` and `src/content/selectors.ts`.
- Run `npm run lint` and `npm test` in the relevant workspace before committing.
