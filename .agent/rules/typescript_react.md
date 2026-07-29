# TypeScript / React Rules

- Functional components with hooks only — no class components.
- `app/`: Tauri commands are invoked via `@tauri-apps/api/core`'s `invoke()`; never call Node/OS APIs directly from React code.
- `extension/`: all VS Code API access goes through `src/extension.ts` or a dedicated module — keep command handlers small and delegate logic to testable functions.
- Run `npm run lint` and `npm test` in the relevant workspace before committing.
