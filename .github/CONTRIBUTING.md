# Contributing to Build-Optimization

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-60%25-green.svg)](https://codecov.io/)

> **Version**: 1.0
> **Last Updated**: 2026-07-29

Thank you for your interest in contributing to Build-Optimization! This document covers everything from code style to the pull-request process across the project's four layers: the C++ backend, the Python middleware, the Tauri + TypeScript frontend, and the IDE/engine integrations.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Development Setup](#2-development-setup)
3. [Code Style Guidelines](#3-code-style-guidelines)
4. [Git Workflow](#4-git-workflow)
5. [Pull Request Process](#5-pull-request-process)
6. [Testing Requirements](#6-testing-requirements)
7. [Documentation Standards](#7-documentation-standards)
8. [Architecture Guidelines](#8-architecture-guidelines)
9. [Adding New Features](#9-adding-new-features)
10. [Issue Reporting](#10-issue-reporting)
11. [Code Review Guidelines](#11-code-review-guidelines)
12. [Community Standards](#12-community-standards)

---

## 1. Getting Started

### 1.1 Prerequisites

- Python 3.9+, Node.js 18+, Rust 1.75+, CMake 3.18+
- [uv](https://github.com/astral-sh/uv), [Pixi](https://pixi.sh/), and [Just](https://github.com/casey/just) installed
- Basic familiarity with knapsack/combinatorial optimization concepts and Hydra configuration

### 1.2 Finding Issues to Work On

1. **Good First Issues** — labeled `good-first-issue`
2. **Help Wanted** — labeled `help-wanted`
3. **Bug Fixes** — labeled `bug`
4. **Feature Requests** — labeled `enhancement`

Issues are triaged onto the [project board](https://github.com/users/ACFHarbinger/projects/15/), split into four views: **C++ Backend + Python Middleware**, **IDE Extension**, **Unreal Engine Plugin**, and **Tauri App**.

### 1.3 Communication Channels

- **GitHub Issues** — bug reports and feature requests
- **GitHub Discussions / Pull Requests** — design discussion and code review

## 2. Development Setup

### 2.1 Fork and Clone

```bash
git clone https://github.com/<you>/Build-Optimization.git
cd Build-Optimization
git remote add upstream https://github.com/ACFHarbinger/Build-Optimization.git
```

### 2.2 Environment Setup

```bash
just setup            # installs Python, C++, and Node dependencies for every module
source .venv/bin/activate
```

Or set up modules individually — see [README.md § Installation & Setup](../README.md#installation--setup).

### 2.3 Pre-commit Hooks

```bash
uv pip install pre-commit
pre-commit install
```

### 2.4 IDE Configuration

VS Code is recommended. Install the Python, rust-analyzer, and ESLint extensions; the repo's `extension/` module itself targets VS Code, so use two windows if you're actively developing it (one for the extension source, one running the Extension Development Host).

## 3. Code Style Guidelines

| Language | Style | Enforced by |
| --- | --- | --- |
| Python | PEP8, 120-char lines, double quotes | `ruff check`, `ruff format` |
| C++ | C++17, `.clang-format` (LLVM base) | `just backend::format` |
| TypeScript | 2-space indent, single quotes | `eslint`, `prettier` |
| Rust | rustfmt defaults | `cargo fmt` |

```bash
# Python
uv run ruff check . --fix
uv run ruff format .
uv run mypy middleware/src

# TypeScript (app/ or extension/)
npm run lint
npm run format
```

Type hints are mandatory on public Python functions; Google-style docstrings are required for public modules, classes, and functions. C++ headers use Doxygen-style comments; TypeScript uses TSDoc.

## 4. Git Workflow

### 4.1 Branch Naming

`<type>/<short-description>`, e.g. `feature/quadratic-knapsack-solver`, `fix/synergy-double-count`.

| Type | Use |
| --- | --- |
| `feature` | New capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | No behavior change |
| `test` | Test-only change |
| `ci` | CI/CD or infra |
| `perf` | Performance improvement |

### 4.2 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <summary>`, e.g. `feat(backend): add branch-and-bound solver for MMKP`.

### 4.3 Keeping Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease
```

## 5. Pull Request Process

### 5.1 Before Opening a PR

- [ ] Tests pass locally for every module you touched
- [ ] Linters/formatters pass
- [ ] Documentation updated (README, `docs/`, or `.moon/ROADMAP.md` as relevant)
- [ ] `.moon/CHANGELOG.md` updated under `Unreleased`

### 5.2 PR Size Guidelines

| Size | Lines changed |
| --- | --- |
| XS | < 50 |
| S | 50–200 |
| M | 200–500 |
| L | 500–1000 |
| XL | > 1000 (split if possible) |

### 5.3 Review Process

CI must pass, at least one approval is required, and review comments must be addressed before merge. Squash-merge is preferred for single-purpose PRs.

## 6. Testing Requirements

| Module | Command | Minimum coverage |
| --- | --- | --- |
| Python middleware | `uv run pytest middleware/tests -v` | 60% |
| C++ backend | `cd backend && pixi run test` | n/a (build must pass) |
| Tauri frontend | `cd app && npm test` | n/a |
| VS Code extension | `cd extension && npm test` | n/a |

Use pytest markers (`slow`, `fast`, `unit`, `integration`, `model`, `data`) to scope test runs — see `pyproject.toml` → `[tool.pytest.ini_options]`.

## 7. Documentation Standards

- Update `README.md` when public commands, directories, or setup steps change.
- Update the relevant file under `docs/` (`ARCHITECTURE.md`, `DEVELOPMENT.md`, `TESTING.md`) when module boundaries or workflows change.
- Update `.moon/ROADMAP.md` and `.moon/CHANGELOG.md` when completing roadmap items.

## 8. Architecture Guidelines

- **Layer separation**: `middleware/src/core` must not import from `middleware/ui`; `backend/` exposes solvers via a stable `pybind11` interface consumed only through `middleware/src/policies`.
- **Adding a new solver**: implement `BaseSolver` in `middleware/src/solvers/`, register a Hydra config under `middleware/configs/policy/`, and add a row to the [README's solver table](../README.md#available-solvers).
- **Adding a C++ export**: add the source under `backend/src/`, declare bindings in `backend/src/bindings.cpp`, and rebuild via `just backend::build-base`.

## 9. Adding New Features

1. Open an issue describing the problem and proposed solution; get maintainer sign-off for anything non-trivial.
2. Implement behind the smallest reasonable surface area — prefer extending an existing config/interface over introducing a new one.
3. Add tests and documentation in the same PR.

## 10. Issue Reporting

**Bug reports** should include: reproduction steps, expected vs. actual behavior, module affected (backend/middleware/app/extension), and environment (OS, Python/Node/Rust versions).

**Feature requests** should include: the problem being solved, a proposed solution, and alternatives considered.

## 11. Code Review Guidelines

**Reviewers**: check correctness, test coverage, and whether the change matches the architecture guidelines above.
**Authors**: respond to every comment, keep PRs focused, and rebase (don't merge-commit) to stay current with `main`.

## 12. Community Standards

Be respectful and constructive. Contributions are credited in `.moon/CHANGELOG.md` and release notes.

---

Contributions are licensed under the same [MIT License](../LICENSE) as the project.
