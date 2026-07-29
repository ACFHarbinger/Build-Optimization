# Skill: Add a New Solver

1. Implement `middleware/src/solvers/<name>.py`, subclassing `BaseSolver` (`middleware/src/solvers/base.py`).
2. Add a Hydra config at `middleware/configs/policy/policy_<name>.yaml`.
3. If a solver alias is needed for the games pipeline, add it to `_SOLVER_ALIAS` in `main.py`.
4. Add unit tests under `middleware/tests/test_solvers.py`.
5. Add a row to the [README's solver table](../../README.md#available-solvers) and mark the corresponding `moon/ROADMAP.md` item done.
