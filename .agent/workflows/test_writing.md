# Workflow: Writing Tests

- Python: pytest, mirror `middleware/src/` package layout under `middleware/tests/`; mark slow tests `@pytest.mark.slow`.
- C++: Catch2 under `backend/tests/` (enable the `dev` Pixi environment).
- TypeScript: Vitest for `app/`, `@vscode/test-cli` for `extension/`.
- Every bug fix gets a regression test before the fix, where practical.

See [`docs/TESTING.md`](../../docs/TESTING.md) for coverage requirements.
