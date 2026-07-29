# Testing

## Per-Module Test Commands

```bash
# Python middleware — unit + integration tests, coverage-gated at 60%
uv run pytest middleware/tests -v --cov=src --cov-report=term-missing

# C++ backend
cd backend && pixi run test

# Tauri frontend
cd app && npm test

# VS Code extension
cd extension && npm test
```

## Markers (Python)

Defined in `middleware/pyproject.toml` → `[tool.pytest.ini_options]`:

| Marker | Use |
| --- | --- |
| `slow` | Long-running tests, excluded from `test-fast` |
| `fast` | Quick unit tests |
| `unit` | Isolated unit tests |
| `integration` | Cross-module tests |
| `model` | Model/solver correctness |
| `data` | Data pipeline / transform tests |

## Coverage

Coverage is enforced at the Python-middleware level (`fail_under = 60` in `middleware/pyproject.toml`) and reported to [Codecov](https://codecov.io/) per [`.github/codecov.yaml`](../.github/codecov.yaml). Patch coverage is currently informational only.

## Writing Tests

- Python: pytest classes named `Test*`, functions named `test_*`; mirror the `middleware/src/` package layout under `middleware/tests/`.
- C++: Catch2, added under `backend/tests/` (enable via the `dev` Pixi environment).
- TypeScript: Vitest for unit/component tests (`app/`), `@vscode/test-cli` for the extension.
