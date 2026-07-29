# Development

## Environment Setup

```bash
just setup             # Python + C++ + Node dependencies for every module
source .venv/bin/activate
```

Per-module setup is documented in [README.md § Installation & Setup](../README.md#installation--setup).

## Day-to-Day Workflow

```bash
just train game=rpg model=am           # Hydra-driven training
just optimize game=rpg policies=sa,ga  # Run the optimization pipeline
just gui                               # Launch the PySide/CLI GUI entry point
just studio                            # Launch the Tauri Studio (Control Tower dashboard)
just lint                              # ruff check
just format                            # ruff format
just test-fast                         # quick unit tests (see tools/test/justfile)
```

Run `just help` for the full categorized command reference, or `just --list` for the raw recipe list.

## Working on Each Module

| Module | Command to iterate |
| --- | --- |
| `middleware/` | `uv run pytest middleware/tests -v` after each change |
| `backend/` | `cd backend && pixi run build && pixi run test` |
| `frontend/` | `cd frontend && npm run tauri:dev` |
| `extension/` | `cd extension && npm run watch:chrome`, then load `extension/dist/chrome` unpacked |

## Debugging

- Python: `uv run python -m pdb main.py ...` or attach your IDE's debugger to the `uv run` process.
- C++: build with `-DCMAKE_BUILD_TYPE=Debug` (edit `backend/pixi.toml`'s `build-base` task) and use `gdb`/`lldb` against the compiled `.so`.
- Tauri: `npm run tauri:dev` opens devtools automatically in debug builds.

See [`docs/TESTING.md`](TESTING.md) for coverage requirements and [`git/CONTRIBUTING.md`](../git/CONTRIBUTING.md) for the full contribution workflow.
