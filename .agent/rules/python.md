# Python Rules

- Type-hint all public function signatures in `middleware/src`.
- Google-style docstrings on public modules, classes, and functions.
- New solvers implement `middleware/src/solvers/base.py`'s `BaseSolver`; register a matching Hydra config under `middleware/configs/policy/`.
- The dashboard lives in `frontend/` (Tauri), not `middleware/` — never add a UI/presentation layer under `middleware/src`.
- Prefer `pathlib.Path` over string paths; prefer `pydantic` models over raw dicts for config-shaped data.
